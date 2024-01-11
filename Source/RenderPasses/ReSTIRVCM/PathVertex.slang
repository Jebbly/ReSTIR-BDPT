import Utils.Sampling.SampleGenerator;
import Utils.Geometry.GeometryHelpers;
import Scene.HitInfo;
import Scene.RaytracingInline;
import LoadShadingData;

///////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// Path vertex interface //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

struct PackedPathVertex {
    float3 mRayOrigin;            ///< Origin of the incoming ray (or the outgoing ray for the first light subpath vertex)
    float mRayDistance;           ///< Hit distance of the incoming ray (or the outgoing ray for the first light subpath vertex)
    float3 mRayDirection;         ///< Direction of the incoming ray (or the outgoing ray for the first light subpath vertex)
    float pad1;
    PackedHitInfo mPackedHit = 0; ///< Packed hit information.

    __init(const Ray ray, const HitInfo hit, const float hitDist) {
        mRayOrigin = ray.origin;
        mRayDistance = hitDist;
        mRayDirection = ray.dir;
        mPackedHit = hit.getData();
    }

    bool IsFinite() {
        return HitInfo(mPackedHit).isValid();
    }
};

struct ReflectanceEvalData {
    float3 mReflectance = 0;
    float mFwdPdfW = 0;
    float mRevPdfW = 0;
};
struct DirectionSampleData {
    float3 mDirection;
    float3 mReflectance = 0;
    float mFwdPdfW = 0;
    float mRevPdfW = 0;
    bool mDeltaLobe = false;

    [mutating]
    void SetReflectance(const ReflectanceEvalData r) {
        mReflectance = r.mReflectance;
        mFwdPdfW = r.mFwdPdfW;
        mRevPdfW = r.mRevPdfW;
    }
};

struct PathVertex {
    PackedPathVertex  mVertex;
    ShadingData       mShadingData;
    IMaterialInstance mMaterial;

    __init(const PackedPathVertex v) {
        mVertex = v;
        if (v.IsFinite()) {
            HitInfo hit = HitInfo(v.mPackedHit);

            let lod = ExplicitLodTextureSampler(0.f);
            mShadingData = loadShadingData(hit, v.mRayOrigin, v.mRayDirection, lod);

            let hints = getMaterialInstanceHints(hit, false);
            mMaterial = gScene.materials.getMaterialInstance(mShadingData, lod, hints);
        }
    }

    PackedPathVertex Pack() {
        return mVertex;
    }

    PathVertex Reversed(const float3 dir) {
        PathVertex vertex;
        vertex.mVertex = mVertex;
        //*
        VertexData v;
        v.coneTexLODValue = 0.f;
        v.curveRadius = mShadingData.curveRadius;
        v.faceNormalW = mShadingData.faceN;
        v.normalW = mShadingData.frame.N;
        v.posW = mShadingData.posW;
        v.tangentW = float4( mShadingData.frame.T, sign(dot(mShadingData.frame.B, cross(mShadingData.frame.N, mShadingData.frame.T))) );
        v.texC = mShadingData.uv;
        let lod = ExplicitLodTextureSampler(0.f);
        vertex.mShadingData = gScene.materials.prepareShadingData(v, mShadingData.materialID, dir, lod);
        /*/
        vertex.mShadingData = mShadingData;
        vertex.mShadingData.V = dir;
        vertex.mShadingData.frontFacing = dot(vertex.mShadingData.faceN, vertex.mShadingData.V) >= 0.f;
        //*/

        HitInfo hit = HitInfo(mVertex.mPackedHit);

        let hints = getMaterialInstanceHints(hit, false);
        vertex.mMaterial = gScene.materials.getMaterialInstance(vertex.mShadingData, lod, hints);

        return vertex;
    }

    bool IsFinite() {
        return mVertex.IsFinite();
    }

    bool IsDelta() {
        if (IsFinite())
            return (mMaterial.getLobeTypes(mShadingData) & uint(LobeType::NonDelta)) == 0;
        else
            return true;
    }

    float3 GetPosition() {
        return mShadingData.posW;
    }

    float3 GetDirIn() {
        return mVertex.mRayDirection;
    }

    bool HasFaceNormal() {
        return IsFinite();
    }
    float3 GetFaceNormal() {
        return IsFinite() ? mShadingData.getOrientedFaceNormal() : 0;
    }

    float GetCosN(const float3 dir) {
        if (IsFinite())
            return abs(dot(mShadingData.faceN, dir));
        else
            return 1;
    }

    float3 ComputeRayOrigin(const float3 dir) {
        return computeRayOrigin(mShadingData.posW, dot(dir, mShadingData.faceN) >= 0.f ? mShadingData.faceN : -mShadingData.faceN);
    }

    BSDFProperties GetBSDFProperties() {
        return mMaterial.getProperties(mShadingData);
    }

    float3 GetEmission() {
        if (IsFinite()) {
            if (dot(mShadingData.faceN, mVertex.mRayDirection) > 0) {
                return 0; // triangle lights are one-sided
            } else {
                return GetBSDFProperties().emission;
            }
        } else {
            return gScene.envMap.eval(mVertex.mRayDirection, 0);
        }
    }

    float GetRoughness() {
        if (IsFinite())
            return GetBSDFProperties().roughness;
        else
            return 0;
    }

    float GetContinuationProbability() {
        return IsDelta() ? 1.f : (1.f - GetBSDFProperties().roughness*.2);
    }

    float GetAdjointCorrection(float3 dirOut) {
        float cosOut = dot(mShadingData.faceN, dirOut);
        float cosIn  = dot(mShadingData.faceN, mShadingData.V);

        float factor = 1.f;

        const BSDFProperties properties = GetBSDFProperties();

        // shading normal correction.
        const float num   = abs(dot(properties.guideNormal, mShadingData.V) * cosOut);
        const float denom = abs(dot(properties.guideNormal, dirOut) * cosIn);
        if (denom >= kMinCosTheta)
            factor = num / denom;

        if (cosIn * cosOut < 0.f) {
            // eta factor correction.
            factor *= 1.f / (properties.eta * properties.eta);
        }

        return factor;
    }

    uint GetLobeTypes() {
        return mMaterial.getLobeTypes(mShadingData);
    }
    bool HasLobe(const LobeType l) {
        return (GetLobeTypes() & uint(l)) != 0;
    }

    ReflectanceEvalData EvaluateReflectance(inout SampleGenerator sg, const float3 dirOut, const bool adjoint) {
        ReflectanceEvalData r;
        r.mReflectance = mMaterial.eval(mShadingData, dirOut, sg);
        if (adjoint) r.mReflectance *= GetAdjointCorrection(dirOut);

        r.mFwdPdfW = mMaterial.evalPdf(mShadingData, dirOut, true);

        const PathVertex revVertex = Reversed(dirOut);
        r.mRevPdfW = revVertex.mMaterial.evalPdf(revVertex.mShadingData, -GetDirIn(), true);

        const float contProb = GetContinuationProbability();
        r.mRevPdfW *= contProb;
        r.mFwdPdfW *= contProb;

        return r;
    }
    DirectionSampleData SampleDirection(inout SampleGenerator sg, const bool adjoint) {
        BSDFSample s;
        DirectionSampleData r = {};
        if (!mMaterial.sample(mShadingData, sg, s, true))
            return r;

        r.mDirection = s.wo;
        r.mReflectance = s.weight;
        r.mDeltaLobe = s.isLobe(LobeType::Delta);
        if (adjoint) r.mReflectance *= GetAdjointCorrection(r.mDirection);

        r.mFwdPdfW = s.pdf;

        if (s.pdf > 0) {
            r.mReflectance *= s.pdf; // multiply pdf out of f*cos*p to get f*cos

            const PathVertex revVertex = Reversed(r.mDirection);
            r.mRevPdfW = revVertex.mMaterial.evalPdf(revVertex.mShadingData, -GetDirIn(), true);
        } else
            r.mRevPdfW = 0;

        const float contProb = GetContinuationProbability();
        r.mFwdPdfW *= contProb;
        r.mRevPdfW *= contProb;

        return r;
    }
};

///////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// Path vertex helpers ///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

PackedPathVertex TraceRay(const Ray ray) {
    HitInfo hit;
    float dist;
    traceSceneRay<true>(ray, hit, dist, 0, 0xFFFFFFFF);
    return PackedPathVertex(ray, hit, dist);
}
bool Occluded(const Ray ray) {
    return !traceSceneVisibilityRay<true>(ray, 0, 0xFFFFFFFF);
}
bool Occluded(const PathVertex vertex, const float3 dir, const float dist) {
    return Occluded(Ray(vertex.ComputeRayOrigin(dir), dir, 0, dist * (1 - 1.0 / 512.0)));
}
