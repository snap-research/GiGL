// Generated by the Scala Plugin for the Protocol Buffer Compiler.
// Do not edit!
//
// Protofile syntax: PROTO3

package snapchat.research.gbml.gigl_resource_config

object GiglResourceConfigProto extends _root_.scalapb.GeneratedFileObject {
  lazy val dependencies: Seq[_root_.scalapb.GeneratedFileObject] = Seq.empty
  lazy val messagesCompanions: Seq[_root_.scalapb.GeneratedMessageCompanion[_ <: _root_.scalapb.GeneratedMessage]] =
    Seq[_root_.scalapb.GeneratedMessageCompanion[_ <: _root_.scalapb.GeneratedMessage]](
      snapchat.research.gbml.gigl_resource_config.SparkResourceConfig,
      snapchat.research.gbml.gigl_resource_config.DataflowResourceConfig,
      snapchat.research.gbml.gigl_resource_config.DataPreprocessorConfig,
      snapchat.research.gbml.gigl_resource_config.VertexAiTrainerConfig,
      snapchat.research.gbml.gigl_resource_config.KFPTrainerConfig,
      snapchat.research.gbml.gigl_resource_config.LocalTrainerConfig,
      snapchat.research.gbml.gigl_resource_config.VertexAiResourceConfig,
      snapchat.research.gbml.gigl_resource_config.KFPResourceConfig,
      snapchat.research.gbml.gigl_resource_config.LocalResourceConfig,
      snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig,
      snapchat.research.gbml.gigl_resource_config.TrainerResourceConfig,
      snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig,
      snapchat.research.gbml.gigl_resource_config.SharedResourceConfig,
      snapchat.research.gbml.gigl_resource_config.GiglResourceConfig
    )
  private lazy val ProtoBytes: _root_.scala.Array[Byte] =
      scalapb.Encoding.fromBase64(scala.collection.immutable.Seq(
  """CjFzbmFwY2hhdC9yZXNlYXJjaC9nYm1sL2dpZ2xfcmVzb3VyY2VfY29uZmlnLnByb3RvEhZzbmFwY2hhdC5yZXNlYXJjaC5nY
  m1sIrgBChNTcGFya1Jlc291cmNlQ29uZmlnEjMKDG1hY2hpbmVfdHlwZRgBIAEoCUIQ4j8NEgttYWNoaW5lVHlwZVILbWFjaGluZ
  VR5cGUSNwoObnVtX2xvY2FsX3NzZHMYAiABKA1CEeI/DhIMbnVtTG9jYWxTc2RzUgxudW1Mb2NhbFNzZHMSMwoMbnVtX3JlcGxpY
  2FzGAMgASgNQhDiPw0SC251bVJlcGxpY2FzUgtudW1SZXBsaWNhcyLuAQoWRGF0YWZsb3dSZXNvdXJjZUNvbmZpZxIwCgtudW1fd
  29ya2VycxgBIAEoDUIP4j8MEgpudW1Xb3JrZXJzUgpudW1Xb3JrZXJzEjoKD21heF9udW1fd29ya2VycxgCIAEoDUIS4j8PEg1tY
  XhOdW1Xb3JrZXJzUg1tYXhOdW1Xb3JrZXJzEjMKDG1hY2hpbmVfdHlwZRgDIAEoCUIQ4j8NEgttYWNoaW5lVHlwZVILbWFjaGluZ
  VR5cGUSMQoMZGlza19zaXplX2diGAQgASgNQg/iPwwSCmRpc2tTaXplR2JSCmRpc2tTaXplR2IiqAIKFkRhdGFQcmVwcm9jZXNzb
  3JDb25maWcShQEKGGVkZ2VfcHJlcHJvY2Vzc29yX2NvbmZpZxgBIAEoCzIuLnNuYXBjaGF0LnJlc2VhcmNoLmdibWwuRGF0YWZsb
  3dSZXNvdXJjZUNvbmZpZ0Ib4j8YEhZlZGdlUHJlcHJvY2Vzc29yQ29uZmlnUhZlZGdlUHJlcHJvY2Vzc29yQ29uZmlnEoUBChhub
  2RlX3ByZXByb2Nlc3Nvcl9jb25maWcYAiABKAsyLi5zbmFwY2hhdC5yZXNlYXJjaC5nYm1sLkRhdGFmbG93UmVzb3VyY2VDb25ma
  WdCG+I/GBIWbm9kZVByZXByb2Nlc3NvckNvbmZpZ1IWbm9kZVByZXByb2Nlc3NvckNvbmZpZyLWAQoVVmVydGV4QWlUcmFpbmVyQ
  29uZmlnEjMKDG1hY2hpbmVfdHlwZRgBIAEoCUIQ4j8NEgttYWNoaW5lVHlwZVILbWFjaGluZVR5cGUSJwoIZ3B1X3R5cGUYAiABK
  AlCDOI/CRIHZ3B1VHlwZVIHZ3B1VHlwZRIqCglncHVfbGltaXQYAyABKA1CDeI/ChIIZ3B1TGltaXRSCGdwdUxpbWl0EjMKDG51b
  V9yZXBsaWNhcxgEIAEoDUIQ4j8NEgtudW1SZXBsaWNhc1ILbnVtUmVwbGljYXMiiQIKEEtGUFRyYWluZXJDb25maWcSMAoLY3B1X
  3JlcXVlc3QYASABKAlCD+I/DBIKY3B1UmVxdWVzdFIKY3B1UmVxdWVzdBI5Cg5tZW1vcnlfcmVxdWVzdBgCIAEoCUIS4j8PEg1tZ
  W1vcnlSZXF1ZXN0Ug1tZW1vcnlSZXF1ZXN0EicKCGdwdV90eXBlGAMgASgJQgziPwkSB2dwdVR5cGVSB2dwdVR5cGUSKgoJZ3B1X
  2xpbWl0GAQgASgNQg3iPwoSCGdwdUxpbWl0UghncHVMaW1pdBIzCgxudW1fcmVwbGljYXMYBSABKA1CEOI/DRILbnVtUmVwbGljY
  XNSC251bVJlcGxpY2FzIkYKEkxvY2FsVHJhaW5lckNvbmZpZxIwCgtudW1fd29ya2VycxgBIAEoDUIP4j8MEgpudW1Xb3JrZXJzU
  gpudW1Xb3JrZXJzIv8BChZWZXJ0ZXhBaVJlc291cmNlQ29uZmlnEjMKDG1hY2hpbmVfdHlwZRgBIAEoCUIQ4j8NEgttYWNoaW5lV
  HlwZVILbWFjaGluZVR5cGUSJwoIZ3B1X3R5cGUYAiABKAlCDOI/CRIHZ3B1VHlwZVIHZ3B1VHlwZRIqCglncHVfbGltaXQYAyABK
  A1CDeI/ChIIZ3B1TGltaXRSCGdwdUxpbWl0EjMKDG51bV9yZXBsaWNhcxgEIAEoDUIQ4j8NEgtudW1SZXBsaWNhc1ILbnVtUmVwb
  GljYXMSJgoHdGltZW91dBgFIAEoDUIM4j8JEgd0aW1lb3V0Ugd0aW1lb3V0IooCChFLRlBSZXNvdXJjZUNvbmZpZxIwCgtjcHVfc
  mVxdWVzdBgBIAEoCUIP4j8MEgpjcHVSZXF1ZXN0UgpjcHVSZXF1ZXN0EjkKDm1lbW9yeV9yZXF1ZXN0GAIgASgJQhLiPw8SDW1lb
  W9yeVJlcXVlc3RSDW1lbW9yeVJlcXVlc3QSJwoIZ3B1X3R5cGUYAyABKAlCDOI/CRIHZ3B1VHlwZVIHZ3B1VHlwZRIqCglncHVfb
  GltaXQYBCABKA1CDeI/ChIIZ3B1TGltaXRSCGdwdUxpbWl0EjMKDG51bV9yZXBsaWNhcxgFIAEoDUIQ4j8NEgtudW1SZXBsaWNhc
  1ILbnVtUmVwbGljYXMiRwoTTG9jYWxSZXNvdXJjZUNvbmZpZxIwCgtudW1fd29ya2VycxgBIAEoDUIP4j8MEgpudW1Xb3JrZXJzU
  gpudW1Xb3JrZXJzIp0DChhEaXN0cmlidXRlZFRyYWluZXJDb25maWcShAEKGHZlcnRleF9haV90cmFpbmVyX2NvbmZpZxgBIAEoC
  zItLnNuYXBjaGF0LnJlc2VhcmNoLmdibWwuVmVydGV4QWlUcmFpbmVyQ29uZmlnQhriPxcSFXZlcnRleEFpVHJhaW5lckNvbmZpZ
  0gAUhV2ZXJ0ZXhBaVRyYWluZXJDb25maWcSbwoSa2ZwX3RyYWluZXJfY29uZmlnGAIgASgLMiguc25hcGNoYXQucmVzZWFyY2guZ
  2JtbC5LRlBUcmFpbmVyQ29uZmlnQhXiPxISEGtmcFRyYWluZXJDb25maWdIAFIQa2ZwVHJhaW5lckNvbmZpZxJ3ChRsb2NhbF90c
  mFpbmVyX2NvbmZpZxgDIAEoCzIqLnNuYXBjaGF0LnJlc2VhcmNoLmdibWwuTG9jYWxUcmFpbmVyQ29uZmlnQhfiPxQSEmxvY2FsV
  HJhaW5lckNvbmZpZ0gAUhJsb2NhbFRyYWluZXJDb25maWdCEAoOdHJhaW5lcl9jb25maWcinQMKFVRyYWluZXJSZXNvdXJjZUNvb
  mZpZxKFAQoYdmVydGV4X2FpX3RyYWluZXJfY29uZmlnGAEgASgLMi4uc25hcGNoYXQucmVzZWFyY2guZ2JtbC5WZXJ0ZXhBaVJlc
  291cmNlQ29uZmlnQhriPxcSFXZlcnRleEFpVHJhaW5lckNvbmZpZ0gAUhV2ZXJ0ZXhBaVRyYWluZXJDb25maWcScAoSa2ZwX3RyY
  WluZXJfY29uZmlnGAIgASgLMikuc25hcGNoYXQucmVzZWFyY2guZ2JtbC5LRlBSZXNvdXJjZUNvbmZpZ0IV4j8SEhBrZnBUcmFpb
  mVyQ29uZmlnSABSEGtmcFRyYWluZXJDb25maWcSeAoUbG9jYWxfdHJhaW5lcl9jb25maWcYAyABKAsyKy5zbmFwY2hhdC5yZXNlY
  XJjaC5nYm1sLkxvY2FsUmVzb3VyY2VDb25maWdCF+I/FBISbG9jYWxUcmFpbmVyQ29uZmlnSABSEmxvY2FsVHJhaW5lckNvbmZpZ
  0IQCg50cmFpbmVyX2NvbmZpZyLUAwoYSW5mZXJlbmNlclJlc291cmNlQ29uZmlnEo4BCht2ZXJ0ZXhfYWlfaW5mZXJlbmNlcl9jb
  25maWcYASABKAsyLi5zbmFwY2hhdC5yZXNlYXJjaC5nYm1sLlZlcnRleEFpUmVzb3VyY2VDb25maWdCHeI/GhIYdmVydGV4QWlJb
  mZlcmVuY2VyQ29uZmlnSABSGHZlcnRleEFpSW5mZXJlbmNlckNvbmZpZxKNAQoaZGF0YWZsb3dfaW5mZXJlbmNlcl9jb25maWcYA
  iABKAsyLi5zbmFwY2hhdC5yZXNlYXJjaC5nYm1sLkRhdGFmbG93UmVzb3VyY2VDb25maWdCHeI/GhIYZGF0YWZsb3dJbmZlcmVuY
  2VyQ29uZmlnSABSGGRhdGFmbG93SW5mZXJlbmNlckNvbmZpZxKBAQoXbG9jYWxfaW5mZXJlbmNlcl9jb25maWcYAyABKAsyKy5zb
  mFwY2hhdC5yZXNlYXJjaC5nYm1sLkxvY2FsUmVzb3VyY2VDb25maWdCGuI/FxIVbG9jYWxJbmZlcmVuY2VyQ29uZmlnSABSFWxvY
  2FsSW5mZXJlbmNlckNvbmZpZ0ITChFpbmZlcmVuY2VyX2NvbmZpZyKXCAoUU2hhcmVkUmVzb3VyY2VDb25maWcSfgoPcmVzb3VyY
  2VfbGFiZWxzGAEgAygLMkAuc25hcGNoYXQucmVzZWFyY2guZ2JtbC5TaGFyZWRSZXNvdXJjZUNvbmZpZy5SZXNvdXJjZUxhYmVsc
  0VudHJ5QhPiPxASDnJlc291cmNlTGFiZWxzUg5yZXNvdXJjZUxhYmVscxKOAQoVY29tbW9uX2NvbXB1dGVfY29uZmlnGAIgASgLM
  kAuc25hcGNoYXQucmVzZWFyY2guZ2JtbC5TaGFyZWRSZXNvdXJjZUNvbmZpZy5Db21tb25Db21wdXRlQ29uZmlnQhjiPxUSE2Nvb
  W1vbkNvbXB1dGVDb25maWdSE2NvbW1vbkNvbXB1dGVDb25maWcalAUKE0NvbW1vbkNvbXB1dGVDb25maWcSJgoHcHJvamVjdBgBI
  AEoCUIM4j8JEgdwcm9qZWN0Ugdwcm9qZWN0EiMKBnJlZ2lvbhgCIAEoCUIL4j8IEgZyZWdpb25SBnJlZ2lvbhJDChJ0ZW1wX2Fzc
  2V0c19idWNrZXQYAyABKAlCFeI/EhIQdGVtcEFzc2V0c0J1Y2tldFIQdGVtcEFzc2V0c0J1Y2tldBJcCht0ZW1wX3JlZ2lvbmFsX
  2Fzc2V0c19idWNrZXQYBCABKAlCHeI/GhIYdGVtcFJlZ2lvbmFsQXNzZXRzQnVja2V0Uhh0ZW1wUmVnaW9uYWxBc3NldHNCdWNrZ
  XQSQwoScGVybV9hc3NldHNfYnVja2V0GAUgASgJQhXiPxISEHBlcm1Bc3NldHNCdWNrZXRSEHBlcm1Bc3NldHNCdWNrZXQSWgobd
  GVtcF9hc3NldHNfYnFfZGF0YXNldF9uYW1lGAYgASgJQhziPxkSF3RlbXBBc3NldHNCcURhdGFzZXROYW1lUhd0ZW1wQXNzZXRzQ
  nFEYXRhc2V0TmFtZRJWChllbWJlZGRpbmdfYnFfZGF0YXNldF9uYW1lGAcgASgJQhviPxgSFmVtYmVkZGluZ0JxRGF0YXNldE5hb
  WVSFmVtYmVkZGluZ0JxRGF0YXNldE5hbWUSVgoZZ2NwX3NlcnZpY2VfYWNjb3VudF9lbWFpbBgIIAEoCUIb4j8YEhZnY3BTZXJ2a
  WNlQWNjb3VudEVtYWlsUhZnY3BTZXJ2aWNlQWNjb3VudEVtYWlsEjwKD2RhdGFmbG93X3J1bm5lchgLIAEoCUIT4j8QEg5kYXRhZ
  mxvd1J1bm5lclIOZGF0YWZsb3dSdW5uZXIaVwoTUmVzb3VyY2VMYWJlbHNFbnRyeRIaCgNrZXkYASABKAlCCOI/BRIDa2V5UgNrZ
  XkSIAoFdmFsdWUYAiABKAlCCuI/BxIFdmFsdWVSBXZhbHVlOgI4ASL3CAoSR2lnbFJlc291cmNlQ29uZmlnElsKGnNoYXJlZF9yZ
  XNvdXJjZV9jb25maWdfdXJpGAEgASgJQhziPxkSF3NoYXJlZFJlc291cmNlQ29uZmlnVXJpSABSF3NoYXJlZFJlc291cmNlQ29uZ
  mlnVXJpEn8KFnNoYXJlZF9yZXNvdXJjZV9jb25maWcYAiABKAsyLC5zbmFwY2hhdC5yZXNlYXJjaC5nYm1sLlNoYXJlZFJlc291c
  mNlQ29uZmlnQhniPxYSFHNoYXJlZFJlc291cmNlQ29uZmlnSABSFHNoYXJlZFJlc291cmNlQ29uZmlnEngKE3ByZXByb2Nlc3Nvc
  l9jb25maWcYDCABKAsyLi5zbmFwY2hhdC5yZXNlYXJjaC5nYm1sLkRhdGFQcmVwcm9jZXNzb3JDb25maWdCF+I/FBIScHJlcHJvY
  2Vzc29yQ29uZmlnUhJwcmVwcm9jZXNzb3JDb25maWcSfwoXc3ViZ3JhcGhfc2FtcGxlcl9jb25maWcYDSABKAsyKy5zbmFwY2hhd
  C5yZXNlYXJjaC5nYm1sLlNwYXJrUmVzb3VyY2VDb25maWdCGuI/FxIVc3ViZ3JhcGhTYW1wbGVyQ29uZmlnUhVzdWJncmFwaFNhb
  XBsZXJDb25maWcSfAoWc3BsaXRfZ2VuZXJhdG9yX2NvbmZpZxgOIAEoCzIrLnNuYXBjaGF0LnJlc2VhcmNoLmdibWwuU3BhcmtSZ
  XNvdXJjZUNvbmZpZ0IZ4j8WEhRzcGxpdEdlbmVyYXRvckNvbmZpZ1IUc3BsaXRHZW5lcmF0b3JDb25maWcSbQoOdHJhaW5lcl9jb
  25maWcYDyABKAsyMC5zbmFwY2hhdC5yZXNlYXJjaC5nYm1sLkRpc3RyaWJ1dGVkVHJhaW5lckNvbmZpZ0IUGAHiPw8SDXRyYWluZ
  XJDb25maWdSDXRyYWluZXJDb25maWcSdAoRaW5mZXJlbmNlcl9jb25maWcYECABKAsyLi5zbmFwY2hhdC5yZXNlYXJjaC5nYm1sL
  kRhdGFmbG93UmVzb3VyY2VDb25maWdCFxgB4j8SEhBpbmZlcmVuY2VyQ29uZmlnUhBpbmZlcmVuY2VyQ29uZmlnEoEBChd0cmFpb
  mVyX3Jlc291cmNlX2NvbmZpZxgRIAEoCzItLnNuYXBjaGF0LnJlc2VhcmNoLmdibWwuVHJhaW5lclJlc291cmNlQ29uZmlnQhriP
  xcSFXRyYWluZXJSZXNvdXJjZUNvbmZpZ1IVdHJhaW5lclJlc291cmNlQ29uZmlnEo0BChppbmZlcmVuY2VyX3Jlc291cmNlX2Nvb
  mZpZxgSIAEoCzIwLnNuYXBjaGF0LnJlc2VhcmNoLmdibWwuSW5mZXJlbmNlclJlc291cmNlQ29uZmlnQh3iPxoSGGluZmVyZW5jZ
  XJSZXNvdXJjZUNvbmZpZ1IYaW5mZXJlbmNlclJlc291cmNlQ29uZmlnQhEKD3NoYXJlZF9yZXNvdXJjZSrjAwoJQ29tcG9uZW50E
  i0KEUNvbXBvbmVudF9Vbmtub3duEAAaFuI/ExIRQ29tcG9uZW50X1Vua25vd24SPwoaQ29tcG9uZW50X0NvbmZpZ19WYWxpZGF0b
  3IQARof4j8cEhpDb21wb25lbnRfQ29uZmlnX1ZhbGlkYXRvchI/ChpDb21wb25lbnRfQ29uZmlnX1BvcHVsYXRvchACGh/iPxwSG
  kNvbXBvbmVudF9Db25maWdfUG9wdWxhdG9yEkEKG0NvbXBvbmVudF9EYXRhX1ByZXByb2Nlc3NvchADGiDiPx0SG0NvbXBvbmVud
  F9EYXRhX1ByZXByb2Nlc3NvchI/ChpDb21wb25lbnRfU3ViZ3JhcGhfU2FtcGxlchAEGh/iPxwSGkNvbXBvbmVudF9TdWJncmFwa
  F9TYW1wbGVyEj0KGUNvbXBvbmVudF9TcGxpdF9HZW5lcmF0b3IQBRoe4j8bEhlDb21wb25lbnRfU3BsaXRfR2VuZXJhdG9yEi0KE
  UNvbXBvbmVudF9UcmFpbmVyEAYaFuI/ExIRQ29tcG9uZW50X1RyYWluZXISMwoUQ29tcG9uZW50X0luZmVyZW5jZXIQBxoZ4j8WE
  hRDb21wb25lbnRfSW5mZXJlbmNlcmIGcHJvdG8z"""
      ).mkString)
  lazy val scalaDescriptor: _root_.scalapb.descriptors.FileDescriptor = {
    val scalaProto = com.google.protobuf.descriptor.FileDescriptorProto.parseFrom(ProtoBytes)
    _root_.scalapb.descriptors.FileDescriptor.buildFrom(scalaProto, dependencies.map(_.scalaDescriptor))
  }
  lazy val javaDescriptor: com.google.protobuf.Descriptors.FileDescriptor = {
    val javaProto = com.google.protobuf.DescriptorProtos.FileDescriptorProto.parseFrom(ProtoBytes)
    com.google.protobuf.Descriptors.FileDescriptor.buildFrom(javaProto, _root_.scala.Array(
    ))
  }
  @deprecated("Use javaDescriptor instead. In a future version this will refer to scalaDescriptor.", "ScalaPB 0.5.47")
  def descriptor: com.google.protobuf.Descriptors.FileDescriptor = javaDescriptor
}