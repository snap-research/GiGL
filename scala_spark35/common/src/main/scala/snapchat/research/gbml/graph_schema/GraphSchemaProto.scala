// Generated by the Scala Plugin for the Protocol Buffer Compiler.
// Do not edit!
//
// Protofile syntax: PROTO3

package snapchat.research.gbml.graph_schema

object GraphSchemaProto extends _root_.scalapb.GeneratedFileObject {
  lazy val dependencies: Seq[_root_.scalapb.GeneratedFileObject] = Seq.empty
  lazy val messagesCompanions: Seq[_root_.scalapb.GeneratedMessageCompanion[_ <: _root_.scalapb.GeneratedMessage]] =
    Seq[_root_.scalapb.GeneratedMessageCompanion[_ <: _root_.scalapb.GeneratedMessage]](
      snapchat.research.gbml.graph_schema.Node,
      snapchat.research.gbml.graph_schema.Edge,
      snapchat.research.gbml.graph_schema.EdgeType,
      snapchat.research.gbml.graph_schema.GraphMetadata,
      snapchat.research.gbml.graph_schema.Graph
    )
  private lazy val ProtoBytes: _root_.scala.Array[Byte] =
      scalapb.Encoding.fromBase64(scala.collection.immutable.Seq(
  """CilzbmFwY2hhdC9yZXNlYXJjaC9nYm1sL2dyYXBoX3NjaGVtYS5wcm90bxIWc25hcGNoYXQucmVzZWFyY2guZ2JtbCLMAQoET
  m9kZRIkCgdub2RlX2lkGAEgASgNQgviPwgSBm5vZGVJZFIGbm9kZUlkEksKE2NvbmRlbnNlZF9ub2RlX3R5cGUYAiABKA1CFuI/E
  xIRY29uZGVuc2VkTm9kZVR5cGVIAFIRY29uZGVuc2VkTm9kZVR5cGWIAQESOQoOZmVhdHVyZV92YWx1ZXMYAyADKAJCEuI/DxINZ
  mVhdHVyZVZhbHVlc1INZmVhdHVyZVZhbHVlc0IWChRfY29uZGVuc2VkX25vZGVfdHlwZSKGAgoERWRnZRIuCgtzcmNfbm9kZV9pZ
  BgBIAEoDUIO4j8LEglzcmNOb2RlSWRSCXNyY05vZGVJZBIuCgtkc3Rfbm9kZV9pZBgCIAEoDUIO4j8LEglkc3ROb2RlSWRSCWRzd
  E5vZGVJZBJLChNjb25kZW5zZWRfZWRnZV90eXBlGAMgASgNQhbiPxMSEWNvbmRlbnNlZEVkZ2VUeXBlSABSEWNvbmRlbnNlZEVkZ
  2VUeXBliAEBEjkKDmZlYXR1cmVfdmFsdWVzGAQgAygCQhLiPw8SDWZlYXR1cmVWYWx1ZXNSDWZlYXR1cmVWYWx1ZXNCFgoUX2Nvb
  mRlbnNlZF9lZGdlX3R5cGUioQEKCEVkZ2VUeXBlEikKCHJlbGF0aW9uGAEgASgJQg3iPwoSCHJlbGF0aW9uUghyZWxhdGlvbhI0C
  g1zcmNfbm9kZV90eXBlGAIgASgJQhDiPw0SC3NyY05vZGVUeXBlUgtzcmNOb2RlVHlwZRI0Cg1kc3Rfbm9kZV90eXBlGAMgASgJQ
  hDiPw0SC2RzdE5vZGVUeXBlUgtkc3ROb2RlVHlwZSKXBQoNR3JhcGhNZXRhZGF0YRItCgpub2RlX3R5cGVzGAEgAygJQg7iPwsSC
  W5vZGVUeXBlc1IJbm9kZVR5cGVzEk8KCmVkZ2VfdHlwZXMYAiADKAsyIC5zbmFwY2hhdC5yZXNlYXJjaC5nYm1sLkVkZ2VUeXBlQ
  g7iPwsSCWVkZ2VUeXBlc1IJZWRnZVR5cGVzEpEBChdjb25kZW5zZWRfZWRnZV90eXBlX21hcBgDIAMoCzI/LnNuYXBjaGF0LnJlc
  2VhcmNoLmdibWwuR3JhcGhNZXRhZGF0YS5Db25kZW5zZWRFZGdlVHlwZU1hcEVudHJ5QhniPxYSFGNvbmRlbnNlZEVkZ2VUeXBlT
  WFwUhRjb25kZW5zZWRFZGdlVHlwZU1hcBKRAQoXY29uZGVuc2VkX25vZGVfdHlwZV9tYXAYBCADKAsyPy5zbmFwY2hhdC5yZXNlY
  XJjaC5nYm1sLkdyYXBoTWV0YWRhdGEuQ29uZGVuc2VkTm9kZVR5cGVNYXBFbnRyeUIZ4j8WEhRjb25kZW5zZWROb2RlVHlwZU1hc
  FIUY29uZGVuc2VkTm9kZVR5cGVNYXAafwoZQ29uZGVuc2VkRWRnZVR5cGVNYXBFbnRyeRIaCgNrZXkYASABKA1CCOI/BRIDa2V5U
  gNrZXkSQgoFdmFsdWUYAiABKAsyIC5zbmFwY2hhdC5yZXNlYXJjaC5nYm1sLkVkZ2VUeXBlQgriPwcSBXZhbHVlUgV2YWx1ZToCO
  AEaXQoZQ29uZGVuc2VkTm9kZVR5cGVNYXBFbnRyeRIaCgNrZXkYASABKA1CCOI/BRIDa2V5UgNrZXkSIAoFdmFsdWUYAiABKAlCC
  uI/BxIFdmFsdWVSBXZhbHVlOgI4ASKHAQoFR3JhcGgSPgoFbm9kZXMYAiADKAsyHC5zbmFwY2hhdC5yZXNlYXJjaC5nYm1sLk5vZ
  GVCCuI/BxIFbm9kZXNSBW5vZGVzEj4KBWVkZ2VzGAMgAygLMhwuc25hcGNoYXQucmVzZWFyY2guZ2JtbC5FZGdlQgriPwcSBWVkZ
  2VzUgVlZGdlc2IGcHJvdG8z"""
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