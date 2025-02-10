// Generated by the Scala Plugin for the Protocol Buffer Compiler.
// Do not edit!
//
// Protofile syntax: PROTO3

package snapchat.research.gbml.gbml_config

object GbmlConfigProto extends _root_.scalapb.GeneratedFileObject {
  lazy val dependencies: Seq[_root_.scalapb.GeneratedFileObject] = Seq(
    snapchat.research.gbml.graph_schema.GraphSchemaProto,
    snapchat.research.gbml.flattened_graph_metadata.FlattenedGraphMetadataProto,
    snapchat.research.gbml.dataset_metadata.DatasetMetadataProto,
    snapchat.research.gbml.trained_model_metadata.TrainedModelMetadataProto,
    snapchat.research.gbml.inference_metadata.InferenceMetadataProto,
    snapchat.research.gbml.postprocessed_metadata.PostprocessedMetadataProto,
    snapchat.research.gbml.subgraph_sampling_strategy.SubgraphSamplingStrategyProto
  )
  lazy val messagesCompanions: Seq[_root_.scalapb.GeneratedMessageCompanion[_ <: _root_.scalapb.GeneratedMessage]] =
    Seq[_root_.scalapb.GeneratedMessageCompanion[_ <: _root_.scalapb.GeneratedMessage]](
      snapchat.research.gbml.gbml_config.GbmlConfig
    )
  private lazy val ProtoBytes: _root_.scala.Array[Byte] =
      scalapb.Encoding.fromBase64(scala.collection.immutable.Seq(
  """CihzbmFwY2hhdC9yZXNlYXJjaC9nYm1sL2dibWxfY29uZmlnLnByb3RvEhZzbmFwY2hhdC5yZXNlYXJjaC5nYm1sGilzbmFwY
  2hhdC9yZXNlYXJjaC9nYm1sL2dyYXBoX3NjaGVtYS5wcm90bxo1c25hcGNoYXQvcmVzZWFyY2gvZ2JtbC9mbGF0dGVuZWRfZ3Jhc
  GhfbWV0YWRhdGEucHJvdG8aLXNuYXBjaGF0L3Jlc2VhcmNoL2dibWwvZGF0YXNldF9tZXRhZGF0YS5wcm90bxozc25hcGNoYXQvc
  mVzZWFyY2gvZ2JtbC90cmFpbmVkX21vZGVsX21ldGFkYXRhLnByb3RvGi9zbmFwY2hhdC9yZXNlYXJjaC9nYm1sL2luZmVyZW5jZ
  V9tZXRhZGF0YS5wcm90bxozc25hcGNoYXQvcmVzZWFyY2gvZ2JtbC9wb3N0cHJvY2Vzc2VkX21ldGFkYXRhLnByb3RvGjdzbmFwY
  2hhdC9yZXNlYXJjaC9nYm1sL3N1YmdyYXBoX3NhbXBsaW5nX3N0cmF0ZWd5LnByb3RvItVHCgpHYm1sQ29uZmlnEmcKDXRhc2tfb
  WV0YWRhdGEYASABKAsyLy5zbmFwY2hhdC5yZXNlYXJjaC5nYm1sLkdibWxDb25maWcuVGFza01ldGFkYXRhQhHiPw4SDHRhc2tNZ
  XRhZGF0YVIMdGFza01ldGFkYXRhEmAKDmdyYXBoX21ldGFkYXRhGAIgASgLMiUuc25hcGNoYXQucmVzZWFyY2guZ2JtbC5HcmFwa
  E1ldGFkYXRhQhLiPw8SDWdyYXBoTWV0YWRhdGFSDWdyYXBoTWV0YWRhdGESZwoNc2hhcmVkX2NvbmZpZxgDIAEoCzIvLnNuYXBja
  GF0LnJlc2VhcmNoLmdibWwuR2JtbENvbmZpZy5TaGFyZWRDb25maWdCEeI/DhIMc2hhcmVkQ29uZmlnUgxzaGFyZWRDb25maWcSa
  woOZGF0YXNldF9jb25maWcYBCABKAsyMC5zbmFwY2hhdC5yZXNlYXJjaC5nYm1sLkdibWxDb25maWcuRGF0YXNldENvbmZpZ0IS4
  j8PEg1kYXRhc2V0Q29uZmlnUg1kYXRhc2V0Q29uZmlnEmsKDnRyYWluZXJfY29uZmlnGAUgASgLMjAuc25hcGNoYXQucmVzZWFyY
  2guZ2JtbC5HYm1sQ29uZmlnLlRyYWluZXJDb25maWdCEuI/DxINdHJhaW5lckNvbmZpZ1INdHJhaW5lckNvbmZpZxJ3ChFpbmZlc
  mVuY2VyX2NvbmZpZxgGIAEoCzIzLnNuYXBjaGF0LnJlc2VhcmNoLmdibWwuR2JtbENvbmZpZy5JbmZlcmVuY2VyQ29uZmlnQhXiP
  xISEGluZmVyZW5jZXJDb25maWdSEGluZmVyZW5jZXJDb25maWcShAEKFXBvc3RfcHJvY2Vzc29yX2NvbmZpZxgJIAEoCzI2LnNuY
  XBjaGF0LnJlc2VhcmNoLmdibWwuR2JtbENvbmZpZy5Qb3N0UHJvY2Vzc29yQ29uZmlnQhjiPxUSE3Bvc3RQcm9jZXNzb3JDb25ma
  WdSE3Bvc3RQcm9jZXNzb3JDb25maWcSawoObWV0cmljc19jb25maWcYByABKAsyMC5zbmFwY2hhdC5yZXNlYXJjaC5nYm1sLkdib
  WxDb25maWcuTWV0cmljc0NvbmZpZ0IS4j8PEg1tZXRyaWNzQ29uZmlnUg1tZXRyaWNzQ29uZmlnEm8KD3Byb2ZpbGVyX2NvbmZpZ
  xgIIAEoCzIxLnNuYXBjaGF0LnJlc2VhcmNoLmdibWwuR2JtbENvbmZpZy5Qcm9maWxlckNvbmZpZ0IT4j8QEg5wcm9maWxlckNvb
  mZpZ1IOcHJvZmlsZXJDb25maWcSbAoNZmVhdHVyZV9mbGFncxgKIAMoCzI0LnNuYXBjaGF0LnJlc2VhcmNoLmdibWwuR2JtbENvb
  mZpZy5GZWF0dXJlRmxhZ3NFbnRyeUIR4j8OEgxmZWF0dXJlRmxhZ3NSDGZlYXR1cmVGbGFncxrnBwoMVGFza01ldGFkYXRhEpwBC
  hhub2RlX2Jhc2VkX3Rhc2tfbWV0YWRhdGEYASABKAsyRS5zbmFwY2hhdC5yZXNlYXJjaC5nYm1sLkdibWxDb25maWcuVGFza01ld
  GFkYXRhLk5vZGVCYXNlZFRhc2tNZXRhZGF0YUIa4j8XEhVub2RlQmFzZWRUYXNrTWV0YWRhdGFIAFIVbm9kZUJhc2VkVGFza01ld
  GFkYXRhEu8BCi9ub2RlX2FuY2hvcl9iYXNlZF9saW5rX3ByZWRpY3Rpb25fdGFza19tZXRhZGF0YRgCIAEoCzJZLnNuYXBjaGF0L
  nJlc2VhcmNoLmdibWwuR2JtbENvbmZpZy5UYXNrTWV0YWRhdGEuTm9kZUFuY2hvckJhc2VkTGlua1ByZWRpY3Rpb25UYXNrTWV0Y
  WRhdGFCLuI/KxIpbm9kZUFuY2hvckJhc2VkTGlua1ByZWRpY3Rpb25UYXNrTWV0YWRhdGFIAFIpbm9kZUFuY2hvckJhc2VkTGlua
  1ByZWRpY3Rpb25UYXNrTWV0YWRhdGESnAEKGGxpbmtfYmFzZWRfdGFza19tZXRhZGF0YRgDIAEoCzJFLnNuYXBjaGF0LnJlc2Vhc
  mNoLmdibWwuR2JtbENvbmZpZy5UYXNrTWV0YWRhdGEuTGlua0Jhc2VkVGFza01ldGFkYXRhQhriPxcSFWxpbmtCYXNlZFRhc2tNZ
  XRhZGF0YUgAUhVsaW5rQmFzZWRUYXNrTWV0YWRhdGEaaAoVTm9kZUJhc2VkVGFza01ldGFkYXRhEk8KFnN1cGVydmlzaW9uX25vZ
  GVfdHlwZXMYASADKAlCGeI/FhIUc3VwZXJ2aXNpb25Ob2RlVHlwZXNSFHN1cGVydmlzaW9uTm9kZVR5cGVzGp4BCilOb2RlQW5ja
  G9yQmFzZWRMaW5rUHJlZGljdGlvblRhc2tNZXRhZGF0YRJxChZzdXBlcnZpc2lvbl9lZGdlX3R5cGVzGAEgAygLMiAuc25hcGNoY
  XQucmVzZWFyY2guZ2JtbC5FZGdlVHlwZUIZ4j8WEhRzdXBlcnZpc2lvbkVkZ2VUeXBlc1IUc3VwZXJ2aXNpb25FZGdlVHlwZXMai
  gEKFUxpbmtCYXNlZFRhc2tNZXRhZGF0YRJxChZzdXBlcnZpc2lvbl9lZGdlX3R5cGVzGAEgAygLMiAuc25hcGNoYXQucmVzZWFyY
  2guZ2JtbC5FZGdlVHlwZUIZ4j8WEhRzdXBlcnZpc2lvbkVkZ2VUeXBlc1IUc3VwZXJ2aXNpb25FZGdlVHlwZXNCDwoNdGFza19tZ
  XRhZGF0YRrYCwoMU2hhcmVkQ29uZmlnElgKGXByZXByb2Nlc3NlZF9tZXRhZGF0YV91cmkYASABKAlCHOI/GRIXcHJlcHJvY2Vzc
  2VkTWV0YWRhdGFVcmlSF3ByZXByb2Nlc3NlZE1ldGFkYXRhVXJpEoUBChhmbGF0dGVuZWRfZ3JhcGhfbWV0YWRhdGEYAiABKAsyL
  i5zbmFwY2hhdC5yZXNlYXJjaC5nYm1sLkZsYXR0ZW5lZEdyYXBoTWV0YWRhdGFCG+I/GBIWZmxhdHRlbmVkR3JhcGhNZXRhZGF0Y
  VIWZmxhdHRlbmVkR3JhcGhNZXRhZGF0YRJoChBkYXRhc2V0X21ldGFkYXRhGAMgASgLMicuc25hcGNoYXQucmVzZWFyY2guZ2Jtb
  C5EYXRhc2V0TWV0YWRhdGFCFOI/ERIPZGF0YXNldE1ldGFkYXRhUg9kYXRhc2V0TWV0YWRhdGESfQoWdHJhaW5lZF9tb2RlbF9tZ
  XRhZGF0YRgEIAEoCzIsLnNuYXBjaGF0LnJlc2VhcmNoLmdibWwuVHJhaW5lZE1vZGVsTWV0YWRhdGFCGeI/FhIUdHJhaW5lZE1vZ
  GVsTWV0YWRhdGFSFHRyYWluZWRNb2RlbE1ldGFkYXRhEnAKEmluZmVyZW5jZV9tZXRhZGF0YRgFIAEoCzIpLnNuYXBjaGF0LnJlc
  2VhcmNoLmdibWwuSW5mZXJlbmNlTWV0YWRhdGFCFuI/ExIRaW5mZXJlbmNlTWV0YWRhdGFSEWluZmVyZW5jZU1ldGFkYXRhEoABC
  hZwb3N0cHJvY2Vzc2VkX21ldGFkYXRhGAwgASgLMi0uc25hcGNoYXQucmVzZWFyY2guZ2JtbC5Qb3N0UHJvY2Vzc2VkTWV0YWRhd
  GFCGuI/FxIVcG9zdHByb2Nlc3NlZE1ldGFkYXRhUhVwb3N0cHJvY2Vzc2VkTWV0YWRhdGEScQoLc2hhcmVkX2FyZ3MYBiADKAsyP
  y5zbmFwY2hhdC5yZXNlYXJjaC5nYm1sLkdibWxDb25maWcuU2hhcmVkQ29uZmlnLlNoYXJlZEFyZ3NFbnRyeUIP4j8MEgpzaGFyZ
  WRBcmdzUgpzaGFyZWRBcmdzEkAKEWlzX2dyYXBoX2RpcmVjdGVkGAcgASgIQhTiPxESD2lzR3JhcGhEaXJlY3RlZFIPaXNHcmFwa
  ERpcmVjdGVkEkkKFHNob3VsZF9za2lwX3RyYWluaW5nGAggASgIQhfiPxQSEnNob3VsZFNraXBUcmFpbmluZ1ISc2hvdWxkU2tpc
  FRyYWluaW5nEn8KKHNob3VsZF9za2lwX2F1dG9tYXRpY190ZW1wX2Fzc2V0X2NsZWFudXAYCSABKAhCKOI/JRIjc2hvdWxkU2tpc
  EF1dG9tYXRpY1RlbXBBc3NldENsZWFudXBSI3Nob3VsZFNraXBBdXRvbWF0aWNUZW1wQXNzZXRDbGVhbnVwEkwKFXNob3VsZF9za
  2lwX2luZmVyZW5jZRgKIAEoCEIY4j8VEhNzaG91bGRTa2lwSW5mZXJlbmNlUhNzaG91bGRTa2lwSW5mZXJlbmNlEl8KHHNob3VsZ
  F9za2lwX21vZGVsX2V2YWx1YXRpb24YCyABKAhCHuI/GxIZc2hvdWxkU2tpcE1vZGVsRXZhbHVhdGlvblIZc2hvdWxkU2tpcE1vZ
  GVsRXZhbHVhdGlvbhKCAQopc2hvdWxkX2luY2x1ZGVfaXNvbGF0ZWRfbm9kZXNfaW5fdHJhaW5pbmcYDSABKAhCKeI/JhIkc2hvd
  WxkSW5jbHVkZUlzb2xhdGVkTm9kZXNJblRyYWluaW5nUiRzaG91bGRJbmNsdWRlSXNvbGF0ZWROb2Rlc0luVHJhaW5pbmcaUwoPU
  2hhcmVkQXJnc0VudHJ5EhoKA2tleRgBIAEoCUII4j8FEgNrZXlSA2tleRIgCgV2YWx1ZRgCIAEoCUIK4j8HEgV2YWx1ZVIFdmFsd
  WU6AjgBGtAUCg1EYXRhc2V0Q29uZmlnEp4BChhkYXRhX3ByZXByb2Nlc3Nvcl9jb25maWcYASABKAsyRy5zbmFwY2hhdC5yZXNlY
  XJjaC5nYm1sLkdibWxDb25maWcuRGF0YXNldENvbmZpZy5EYXRhUHJlcHJvY2Vzc29yQ29uZmlnQhviPxgSFmRhdGFQcmVwcm9jZ
  XNzb3JDb25maWdSFmRhdGFQcmVwcm9jZXNzb3JDb25maWcSmgEKF3N1YmdyYXBoX3NhbXBsZXJfY29uZmlnGAIgASgLMkYuc25hc
  GNoYXQucmVzZWFyY2guZ2JtbC5HYm1sQ29uZmlnLkRhdGFzZXRDb25maWcuU3ViZ3JhcGhTYW1wbGVyQ29uZmlnQhriPxcSFXN1Y
  mdyYXBoU2FtcGxlckNvbmZpZ1IVc3ViZ3JhcGhTYW1wbGVyQ29uZmlnEpYBChZzcGxpdF9nZW5lcmF0b3JfY29uZmlnGAMgASgLM
  kUuc25hcGNoYXQucmVzZWFyY2guZ2JtbC5HYm1sQ29uZmlnLkRhdGFzZXRDb25maWcuU3BsaXRHZW5lcmF0b3JDb25maWdCGeI/F
  hIUc3BsaXRHZW5lcmF0b3JDb25maWdSFHNwbGl0R2VuZXJhdG9yQ29uZmlnGpoDChZEYXRhUHJlcHJvY2Vzc29yQ29uZmlnEmwKI
  WRhdGFfcHJlcHJvY2Vzc29yX2NvbmZpZ19jbHNfcGF0aBgBIAEoCUIi4j8fEh1kYXRhUHJlcHJvY2Vzc29yQ29uZmlnQ2xzUGF0a
  FIdZGF0YVByZXByb2Nlc3NvckNvbmZpZ0Nsc1BhdGgSsgEKFmRhdGFfcHJlcHJvY2Vzc29yX2FyZ3MYAiADKAsyYS5zbmFwY2hhd
  C5yZXNlYXJjaC5nYm1sLkdibWxDb25maWcuRGF0YXNldENvbmZpZy5EYXRhUHJlcHJvY2Vzc29yQ29uZmlnLkRhdGFQcmVwcm9jZ
  XNzb3JBcmdzRW50cnlCGeI/FhIUZGF0YVByZXByb2Nlc3NvckFyZ3NSFGRhdGFQcmVwcm9jZXNzb3JBcmdzGl0KGURhdGFQcmVwc
  m9jZXNzb3JBcmdzRW50cnkSGgoDa2V5GAEgASgJQgjiPwUSA2tleVIDa2V5EiAKBXZhbHVlGAIgASgJQgriPwcSBXZhbHVlUgV2Y
  Wx1ZToCOAEasQgKFVN1YmdyYXBoU2FtcGxlckNvbmZpZxIpCghudW1faG9wcxgBIAEoDUIOGAHiPwkSB251bUhvcHNSB251bUhvc
  HMSUgoXbnVtX25laWdoYm9yc190b19zYW1wbGUYAiABKAVCGxgB4j8WEhRudW1OZWlnaGJvcnNUb1NhbXBsZVIUbnVtTmVpZ2hib
  3JzVG9TYW1wbGUSjQEKGnN1YmdyYXBoX3NhbXBsaW5nX3N0cmF0ZWd5GAogASgLMjAuc25hcGNoYXQucmVzZWFyY2guZ2JtbC5Td
  WJncmFwaFNhbXBsaW5nU3RyYXRlZ3lCHeI/GhIYc3ViZ3JhcGhTYW1wbGluZ1N0cmF0ZWd5UhhzdWJncmFwaFNhbXBsaW5nU3RyY
  XRlZ3kSSQoUbnVtX3Bvc2l0aXZlX3NhbXBsZXMYAyABKA1CF+I/FBISbnVtUG9zaXRpdmVTYW1wbGVzUhJudW1Qb3NpdGl2ZVNhb
  XBsZXMSpAEKEmV4cGVyaW1lbnRhbF9mbGFncxgFIAMoCzJdLnNuYXBjaGF0LnJlc2VhcmNoLmdibWwuR2JtbENvbmZpZy5EYXRhc
  2V0Q29uZmlnLlN1YmdyYXBoU2FtcGxlckNvbmZpZy5FeHBlcmltZW50YWxGbGFnc0VudHJ5QhbiPxMSEWV4cGVyaW1lbnRhbEZsY
  WdzUhFleHBlcmltZW50YWxGbGFncxJtCiJudW1fbWF4X3RyYWluaW5nX3NhbXBsZXNfdG9fb3V0cHV0GAYgASgNQiLiPx8SHW51b
  U1heFRyYWluaW5nU2FtcGxlc1RvT3V0cHV0Uh1udW1NYXhUcmFpbmluZ1NhbXBsZXNUb091dHB1dBJuCiFudW1fdXNlcl9kZWZpb
  mVkX3Bvc2l0aXZlX3NhbXBsZXMYByABKA1CJBgB4j8fEh1udW1Vc2VyRGVmaW5lZFBvc2l0aXZlU2FtcGxlc1IdbnVtVXNlckRlZ
  mluZWRQb3NpdGl2ZVNhbXBsZXMSbgohbnVtX3VzZXJfZGVmaW5lZF9uZWdhdGl2ZV9zYW1wbGVzGAggASgNQiQYAeI/HxIdbnVtV
  XNlckRlZmluZWROZWdhdGl2ZVNhbXBsZXNSHW51bVVzZXJEZWZpbmVkTmVnYXRpdmVTYW1wbGVzEmwKD2dyYXBoX2RiX2NvbmZpZ
  xgJIAEoCzIwLnNuYXBjaGF0LnJlc2VhcmNoLmdibWwuR2JtbENvbmZpZy5HcmFwaERCQ29uZmlnQhLiPw8SDWdyYXBoRGJDb25ma
  WdSDWdyYXBoRGJDb25maWcaWgoWRXhwZXJpbWVudGFsRmxhZ3NFbnRyeRIaCgNrZXkYASABKAlCCOI/BRIDa2V5UgNrZXkSIAoFd
  mFsdWUYAiABKAlCCuI/BxIFdmFsdWVSBXZhbHVlOgI4ARqWBQoUU3BsaXRHZW5lcmF0b3JDb25maWcSUAoXc3BsaXRfc3RyYXRlZ
  3lfY2xzX3BhdGgYASABKAlCGeI/FhIUc3BsaXRTdHJhdGVneUNsc1BhdGhSFHNwbGl0U3RyYXRlZ3lDbHNQYXRoEqQBChNzcGxpd
  F9zdHJhdGVneV9hcmdzGAIgAygLMlwuc25hcGNoYXQucmVzZWFyY2guZ2JtbC5HYm1sQ29uZmlnLkRhdGFzZXRDb25maWcuU3Bsa
  XRHZW5lcmF0b3JDb25maWcuU3BsaXRTdHJhdGVneUFyZ3NFbnRyeUIW4j8TEhFzcGxpdFN0cmF0ZWd5QXJnc1IRc3BsaXRTdHJhd
  GVneUFyZ3MSQAoRYXNzaWduZXJfY2xzX3BhdGgYAyABKAlCFOI/ERIPYXNzaWduZXJDbHNQYXRoUg9hc3NpZ25lckNsc1BhdGgSj
  wEKDWFzc2lnbmVyX2FyZ3MYBCADKAsyVy5zbmFwY2hhdC5yZXNlYXJjaC5nYm1sLkdibWxDb25maWcuRGF0YXNldENvbmZpZy5Tc
  GxpdEdlbmVyYXRvckNvbmZpZy5Bc3NpZ25lckFyZ3NFbnRyeUIR4j8OEgxhc3NpZ25lckFyZ3NSDGFzc2lnbmVyQXJncxpaChZTc
  GxpdFN0cmF0ZWd5QXJnc0VudHJ5EhoKA2tleRgBIAEoCUII4j8FEgNrZXlSA2tleRIgCgV2YWx1ZRgCIAEoCUIK4j8HEgV2YWx1Z
  VIFdmFsdWU6AjgBGlUKEUFzc2lnbmVyQXJnc0VudHJ5EhoKA2tleRgBIAEoCUII4j8FEgNrZXlSA2tleRIgCgV2YWx1ZRgCIAEoC
  UIK4j8HEgV2YWx1ZVIFdmFsdWU6AjgBGsMGCg1HcmFwaERCQ29uZmlnEloKG2dyYXBoX2RiX2luZ2VzdGlvbl9jbHNfcGF0aBgBI
  AEoCUIc4j8ZEhdncmFwaERiSW5nZXN0aW9uQ2xzUGF0aFIXZ3JhcGhEYkluZ2VzdGlvbkNsc1BhdGgSnAEKF2dyYXBoX2RiX2luZ
  2VzdGlvbl9hcmdzGAIgAygLMkouc25hcGNoYXQucmVzZWFyY2guZ2JtbC5HYm1sQ29uZmlnLkdyYXBoREJDb25maWcuR3JhcGhEY
  kluZ2VzdGlvbkFyZ3NFbnRyeUIZ4j8WEhRncmFwaERiSW5nZXN0aW9uQXJnc1IUZ3JhcGhEYkluZ2VzdGlvbkFyZ3MSdwoNZ3Jhc
  GhfZGJfYXJncxgDIAMoCzJBLnNuYXBjaGF0LnJlc2VhcmNoLmdibWwuR2JtbENvbmZpZy5HcmFwaERCQ29uZmlnLkdyYXBoRGJBc
  mdzRW50cnlCEOI/DRILZ3JhcGhEYkFyZ3NSC2dyYXBoRGJBcmdzEpcBChdncmFwaF9kYl9zYW1wbGVyX2NvbmZpZxgEIAEoCzJFL
  nNuYXBjaGF0LnJlc2VhcmNoLmdibWwuR2JtbENvbmZpZy5HcmFwaERCQ29uZmlnLkdyYXBoREJTZXJ2aWNlQ29uZmlnQhniPxYSF
  GdyYXBoRGJTYW1wbGVyQ29uZmlnUhRncmFwaERiU2FtcGxlckNvbmZpZxpdChlHcmFwaERiSW5nZXN0aW9uQXJnc0VudHJ5EhoKA
  2tleRgBIAEoCUII4j8FEgNrZXlSA2tleRIgCgV2YWx1ZRgCIAEoCUIK4j8HEgV2YWx1ZVIFdmFsdWU6AjgBGlQKEEdyYXBoRGJBc
  mdzRW50cnkSGgoDa2V5GAEgASgJQgjiPwUSA2tleVIDa2V5EiAKBXZhbHVlGAIgASgJQgriPwcSBXZhbHVlUgV2YWx1ZToCOAEab
  woUR3JhcGhEQlNlcnZpY2VDb25maWcSVwoaZ3JhcGhfZGJfY2xpZW50X2NsYXNzX3BhdGgYASABKAlCG+I/GBIWZ3JhcGhEYkNsa
  WVudENsYXNzUGF0aFIWZ3JhcGhEYkNsaWVudENsYXNzUGF0aBrXAwoNVHJhaW5lckNvbmZpZxI9ChB0cmFpbmVyX2Nsc19wYXRoG
  AEgASgJQhPiPxASDnRyYWluZXJDbHNQYXRoUg50cmFpbmVyQ2xzUGF0aBJ2Cgx0cmFpbmVyX2FyZ3MYAiADKAsyQS5zbmFwY2hhd
  C5yZXNlYXJjaC5nYm1sLkdibWxDb25maWcuVHJhaW5lckNvbmZpZy5UcmFpbmVyQXJnc0VudHJ5QhDiPw0SC3RyYWluZXJBcmdzU
  gt0cmFpbmVyQXJncxIpCghjbHNfcGF0aBhkIAEoCUIM4j8JEgdjbHNQYXRoSABSB2Nsc1BhdGgSKAoHY29tbWFuZBhlIAEoCUIM4
  j8JEgdjb21tYW5kSABSB2NvbW1hbmQSVgoZc2hvdWxkX2xvZ190b190ZW5zb3Jib2FyZBgMIAEoCEIb4j8YEhZzaG91bGRMb2dUb
  1RlbnNvcmJvYXJkUhZzaG91bGRMb2dUb1RlbnNvcmJvYXJkGlQKEFRyYWluZXJBcmdzRW50cnkSGgoDa2V5GAEgASgJQgjiPwUSA
  2tleVIDa2V5EiAKBXZhbHVlGAIgASgJQgriPwcSBXZhbHVlUgV2YWx1ZToCOAFCDAoKZXhlY3V0YWJsZRrpAwoQSW5mZXJlbmNlc
  kNvbmZpZxKFAQoPaW5mZXJlbmNlcl9hcmdzGAEgAygLMkcuc25hcGNoYXQucmVzZWFyY2guZ2JtbC5HYm1sQ29uZmlnLkluZmVyZ
  W5jZXJDb25maWcuSW5mZXJlbmNlckFyZ3NFbnRyeUIT4j8QEg5pbmZlcmVuY2VyQXJnc1IOaW5mZXJlbmNlckFyZ3MSRgoTaW5mZ
  XJlbmNlcl9jbHNfcGF0aBgCIAEoCUIW4j8TEhFpbmZlcmVuY2VyQ2xzUGF0aFIRaW5mZXJlbmNlckNsc1BhdGgSKQoIY2xzX3Bhd
  GgYZCABKAlCDOI/CRIHY2xzUGF0aEgAUgdjbHNQYXRoEigKB2NvbW1hbmQYZSABKAlCDOI/CRIHY29tbWFuZEgAUgdjb21tYW5kE
  kkKFGluZmVyZW5jZV9iYXRjaF9zaXplGAUgASgNQhfiPxQSEmluZmVyZW5jZUJhdGNoU2l6ZVISaW5mZXJlbmNlQmF0Y2hTaXplG
  lcKE0luZmVyZW5jZXJBcmdzRW50cnkSGgoDa2V5GAEgASgJQgjiPwUSA2tleVIDa2V5EiAKBXZhbHVlGAIgASgJQgriPwcSBXZhb
  HVlUgV2YWx1ZToCOAFCDAoKZXhlY3V0YWJsZRrbAgoTUG9zdFByb2Nlc3NvckNvbmZpZxKVAQoTcG9zdF9wcm9jZXNzb3JfYXJnc
  xgBIAMoCzJNLnNuYXBjaGF0LnJlc2VhcmNoLmdibWwuR2JtbENvbmZpZy5Qb3N0UHJvY2Vzc29yQ29uZmlnLlBvc3RQcm9jZXNzb
  3JBcmdzRW50cnlCFuI/ExIRcG9zdFByb2Nlc3NvckFyZ3NSEXBvc3RQcm9jZXNzb3JBcmdzElAKF3Bvc3RfcHJvY2Vzc29yX2Nsc
  19wYXRoGAIgASgJQhniPxYSFHBvc3RQcm9jZXNzb3JDbHNQYXRoUhRwb3N0UHJvY2Vzc29yQ2xzUGF0aBpaChZQb3N0UHJvY2Vzc
  29yQXJnc0VudHJ5EhoKA2tleRgBIAEoCUII4j8FEgNrZXlSA2tleRIgCgV2YWx1ZRgCIAEoCUIK4j8HEgV2YWx1ZVIFdmFsdWU6A
  jgBGpwCCg1NZXRyaWNzQ29uZmlnEj0KEG1ldHJpY3NfY2xzX3BhdGgYASABKAlCE+I/EBIObWV0cmljc0Nsc1BhdGhSDm1ldHJpY
  3NDbHNQYXRoEnYKDG1ldHJpY3NfYXJncxgCIAMoCzJBLnNuYXBjaGF0LnJlc2VhcmNoLmdibWwuR2JtbENvbmZpZy5NZXRyaWNzQ
  29uZmlnLk1ldHJpY3NBcmdzRW50cnlCEOI/DRILbWV0cmljc0FyZ3NSC21ldHJpY3NBcmdzGlQKEE1ldHJpY3NBcmdzRW50cnkSG
  goDa2V5GAEgASgJQgjiPwUSA2tleVIDa2V5EiAKBXZhbHVlGAIgASgJQgriPwcSBXZhbHVlUgV2YWx1ZToCOAEa9AIKDlByb2Zpb
  GVyQ29uZmlnEk8KFnNob3VsZF9lbmFibGVfcHJvZmlsZXIYASABKAhCGeI/FhIUc2hvdWxkRW5hYmxlUHJvZmlsZXJSFHNob3VsZ
  EVuYWJsZVByb2ZpbGVyEj0KEHByb2ZpbGVyX2xvZ19kaXIYAiABKAlCE+I/EBIOcHJvZmlsZXJMb2dEaXJSDnByb2ZpbGVyTG9nR
  GlyEnsKDXByb2ZpbGVyX2FyZ3MYAyADKAsyQy5zbmFwY2hhdC5yZXNlYXJjaC5nYm1sLkdibWxDb25maWcuUHJvZmlsZXJDb25ma
  WcuUHJvZmlsZXJBcmdzRW50cnlCEeI/DhIMcHJvZmlsZXJBcmdzUgxwcm9maWxlckFyZ3MaVQoRUHJvZmlsZXJBcmdzRW50cnkSG
  goDa2V5GAEgASgJQgjiPwUSA2tleVIDa2V5EiAKBXZhbHVlGAIgASgJQgriPwcSBXZhbHVlUgV2YWx1ZToCOAEaVQoRRmVhdHVyZ
  UZsYWdzRW50cnkSGgoDa2V5GAEgASgJQgjiPwUSA2tleVIDa2V5EiAKBXZhbHVlGAIgASgJQgriPwcSBXZhbHVlUgV2YWx1ZToCO
  AFiBnByb3RvMw=="""
      ).mkString)
  lazy val scalaDescriptor: _root_.scalapb.descriptors.FileDescriptor = {
    val scalaProto = com.google.protobuf.descriptor.FileDescriptorProto.parseFrom(ProtoBytes)
    _root_.scalapb.descriptors.FileDescriptor.buildFrom(scalaProto, dependencies.map(_.scalaDescriptor))
  }
  lazy val javaDescriptor: com.google.protobuf.Descriptors.FileDescriptor = {
    val javaProto = com.google.protobuf.DescriptorProtos.FileDescriptorProto.parseFrom(ProtoBytes)
    com.google.protobuf.Descriptors.FileDescriptor.buildFrom(javaProto, _root_.scala.Array(
      snapchat.research.gbml.graph_schema.GraphSchemaProto.javaDescriptor,
      snapchat.research.gbml.flattened_graph_metadata.FlattenedGraphMetadataProto.javaDescriptor,
      snapchat.research.gbml.dataset_metadata.DatasetMetadataProto.javaDescriptor,
      snapchat.research.gbml.trained_model_metadata.TrainedModelMetadataProto.javaDescriptor,
      snapchat.research.gbml.inference_metadata.InferenceMetadataProto.javaDescriptor,
      snapchat.research.gbml.postprocessed_metadata.PostprocessedMetadataProto.javaDescriptor,
      snapchat.research.gbml.subgraph_sampling_strategy.SubgraphSamplingStrategyProto.javaDescriptor
    ))
  }
  @deprecated("Use javaDescriptor instead. In a future version this will refer to scalaDescriptor.", "ScalaPB 0.5.47")
  def descriptor: com.google.protobuf.Descriptors.FileDescriptor = javaDescriptor
}