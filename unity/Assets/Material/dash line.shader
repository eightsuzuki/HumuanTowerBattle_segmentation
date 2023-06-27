Shader "Unlit/DashLine"
{
   Properties {
       _MainTex ("Texture", 2D) = "white" {}
       _Color ("Color", Color) = (1,1,1,1)
       _Rate ("Rate", Range(0, 1)) = 0.5
   }
   SubShader {
       Tags { "RenderType"="Transparent" "Queue" = "Transparent" }
       Blend SrcAlpha OneMinusSrcAlpha 
       LOD 100

       Pass {
           CGPROGRAM
           #pragma vertex vert
           #pragma fragment frag

           #include "UnityCG.cginc"

           struct appdata {
               float4 vertex : POSITION;
               float2 uv : TEXCOORD0;
           };

           struct v2f {
               float2 uv : TEXCOORD0;
               float4 vertex : SV_POSITION;
           };

           sampler2D _MainTex;
           float4 _MainTex_ST;
           float _Rate;
           fixed4 _Color;

           v2f vert (appdata v) {
               v2f o;
               o.vertex = UnityObjectToClipPos(v.vertex);
               o.uv = TRANSFORM_TEX(v.uv, _MainTex);
               return o;
           }

           fixed4 frag (v2f i) : SV_Target {
               return step(_Rate, frac(i.uv.x)) * _Color;
           }
           ENDCG
       }
   }
}