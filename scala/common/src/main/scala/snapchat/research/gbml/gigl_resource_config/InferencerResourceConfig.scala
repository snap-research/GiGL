// Generated by the Scala Plugin for the Protocol Buffer Compiler.
// Do not edit!
//
// Protofile syntax: PROTO3

package snapchat.research.gbml.gigl_resource_config

/** Configuration for distributed inference resources
  */
@SerialVersionUID(0L)
final case class InferencerResourceConfig(
    inferencerConfig: snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.InferencerConfig = snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.InferencerConfig.Empty,
    unknownFields: _root_.scalapb.UnknownFieldSet = _root_.scalapb.UnknownFieldSet.empty
    ) extends scalapb.GeneratedMessage with scalapb.lenses.Updatable[InferencerResourceConfig] {
    @transient
    private[this] var __serializedSizeMemoized: _root_.scala.Int = 0
    private[this] def __computeSerializedSize(): _root_.scala.Int = {
      var __size = 0
      if (inferencerConfig.vertexAiInferencerConfig.isDefined) {
        val __value = inferencerConfig.vertexAiInferencerConfig.get
        __size += 1 + _root_.com.google.protobuf.CodedOutputStream.computeUInt32SizeNoTag(__value.serializedSize) + __value.serializedSize
      };
      if (inferencerConfig.dataflowInferencerConfig.isDefined) {
        val __value = inferencerConfig.dataflowInferencerConfig.get
        __size += 1 + _root_.com.google.protobuf.CodedOutputStream.computeUInt32SizeNoTag(__value.serializedSize) + __value.serializedSize
      };
      if (inferencerConfig.localInferencerConfig.isDefined) {
        val __value = inferencerConfig.localInferencerConfig.get
        __size += 1 + _root_.com.google.protobuf.CodedOutputStream.computeUInt32SizeNoTag(__value.serializedSize) + __value.serializedSize
      };
      __size += unknownFields.serializedSize
      __size
    }
    override def serializedSize: _root_.scala.Int = {
      var __size = __serializedSizeMemoized
      if (__size == 0) {
        __size = __computeSerializedSize() + 1
        __serializedSizeMemoized = __size
      }
      __size - 1
      
    }
    def writeTo(`_output__`: _root_.com.google.protobuf.CodedOutputStream): _root_.scala.Unit = {
      inferencerConfig.vertexAiInferencerConfig.foreach { __v =>
        val __m = __v
        _output__.writeTag(1, 2)
        _output__.writeUInt32NoTag(__m.serializedSize)
        __m.writeTo(_output__)
      };
      inferencerConfig.dataflowInferencerConfig.foreach { __v =>
        val __m = __v
        _output__.writeTag(2, 2)
        _output__.writeUInt32NoTag(__m.serializedSize)
        __m.writeTo(_output__)
      };
      inferencerConfig.localInferencerConfig.foreach { __v =>
        val __m = __v
        _output__.writeTag(3, 2)
        _output__.writeUInt32NoTag(__m.serializedSize)
        __m.writeTo(_output__)
      };
      unknownFields.writeTo(_output__)
    }
    def getVertexAiInferencerConfig: snapchat.research.gbml.gigl_resource_config.VertexAiResourceConfig = inferencerConfig.vertexAiInferencerConfig.getOrElse(snapchat.research.gbml.gigl_resource_config.VertexAiResourceConfig.defaultInstance)
    def withVertexAiInferencerConfig(__v: snapchat.research.gbml.gigl_resource_config.VertexAiResourceConfig): InferencerResourceConfig = copy(inferencerConfig = snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.InferencerConfig.VertexAiInferencerConfig(__v))
    def getDataflowInferencerConfig: snapchat.research.gbml.gigl_resource_config.DataflowResourceConfig = inferencerConfig.dataflowInferencerConfig.getOrElse(snapchat.research.gbml.gigl_resource_config.DataflowResourceConfig.defaultInstance)
    def withDataflowInferencerConfig(__v: snapchat.research.gbml.gigl_resource_config.DataflowResourceConfig): InferencerResourceConfig = copy(inferencerConfig = snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.InferencerConfig.DataflowInferencerConfig(__v))
    def getLocalInferencerConfig: snapchat.research.gbml.gigl_resource_config.LocalResourceConfig = inferencerConfig.localInferencerConfig.getOrElse(snapchat.research.gbml.gigl_resource_config.LocalResourceConfig.defaultInstance)
    def withLocalInferencerConfig(__v: snapchat.research.gbml.gigl_resource_config.LocalResourceConfig): InferencerResourceConfig = copy(inferencerConfig = snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.InferencerConfig.LocalInferencerConfig(__v))
    def clearInferencerConfig: InferencerResourceConfig = copy(inferencerConfig = snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.InferencerConfig.Empty)
    def withInferencerConfig(__v: snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.InferencerConfig): InferencerResourceConfig = copy(inferencerConfig = __v)
    def withUnknownFields(__v: _root_.scalapb.UnknownFieldSet) = copy(unknownFields = __v)
    def discardUnknownFields = copy(unknownFields = _root_.scalapb.UnknownFieldSet.empty)
    def getFieldByNumber(__fieldNumber: _root_.scala.Int): _root_.scala.Any = {
      (__fieldNumber: @_root_.scala.unchecked) match {
        case 1 => inferencerConfig.vertexAiInferencerConfig.orNull
        case 2 => inferencerConfig.dataflowInferencerConfig.orNull
        case 3 => inferencerConfig.localInferencerConfig.orNull
      }
    }
    def getField(__field: _root_.scalapb.descriptors.FieldDescriptor): _root_.scalapb.descriptors.PValue = {
      _root_.scala.Predef.require(__field.containingMessage eq companion.scalaDescriptor)
      (__field.number: @_root_.scala.unchecked) match {
        case 1 => inferencerConfig.vertexAiInferencerConfig.map(_.toPMessage).getOrElse(_root_.scalapb.descriptors.PEmpty)
        case 2 => inferencerConfig.dataflowInferencerConfig.map(_.toPMessage).getOrElse(_root_.scalapb.descriptors.PEmpty)
        case 3 => inferencerConfig.localInferencerConfig.map(_.toPMessage).getOrElse(_root_.scalapb.descriptors.PEmpty)
      }
    }
    def toProtoString: _root_.scala.Predef.String = _root_.scalapb.TextFormat.printToUnicodeString(this)
    def companion: snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.type = snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig
    // @@protoc_insertion_point(GeneratedMessage[snapchat.research.gbml.InferencerResourceConfig])
}

object InferencerResourceConfig extends scalapb.GeneratedMessageCompanion[snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig] {
  implicit def messageCompanion: scalapb.GeneratedMessageCompanion[snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig] = this
  def parseFrom(`_input__`: _root_.com.google.protobuf.CodedInputStream): snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig = {
    var __inferencerConfig: snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.InferencerConfig = snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.InferencerConfig.Empty
    var `_unknownFields__`: _root_.scalapb.UnknownFieldSet.Builder = null
    var _done__ = false
    while (!_done__) {
      val _tag__ = _input__.readTag()
      _tag__ match {
        case 0 => _done__ = true
        case 10 =>
          __inferencerConfig = snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.InferencerConfig.VertexAiInferencerConfig(__inferencerConfig.vertexAiInferencerConfig.fold(_root_.scalapb.LiteParser.readMessage[snapchat.research.gbml.gigl_resource_config.VertexAiResourceConfig](_input__))(_root_.scalapb.LiteParser.readMessage(_input__, _)))
        case 18 =>
          __inferencerConfig = snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.InferencerConfig.DataflowInferencerConfig(__inferencerConfig.dataflowInferencerConfig.fold(_root_.scalapb.LiteParser.readMessage[snapchat.research.gbml.gigl_resource_config.DataflowResourceConfig](_input__))(_root_.scalapb.LiteParser.readMessage(_input__, _)))
        case 26 =>
          __inferencerConfig = snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.InferencerConfig.LocalInferencerConfig(__inferencerConfig.localInferencerConfig.fold(_root_.scalapb.LiteParser.readMessage[snapchat.research.gbml.gigl_resource_config.LocalResourceConfig](_input__))(_root_.scalapb.LiteParser.readMessage(_input__, _)))
        case tag =>
          if (_unknownFields__ == null) {
            _unknownFields__ = new _root_.scalapb.UnknownFieldSet.Builder()
          }
          _unknownFields__.parseField(tag, _input__)
      }
    }
    snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig(
        inferencerConfig = __inferencerConfig,
        unknownFields = if (_unknownFields__ == null) _root_.scalapb.UnknownFieldSet.empty else _unknownFields__.result()
    )
  }
  implicit def messageReads: _root_.scalapb.descriptors.Reads[snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig] = _root_.scalapb.descriptors.Reads{
    case _root_.scalapb.descriptors.PMessage(__fieldsMap) =>
      _root_.scala.Predef.require(__fieldsMap.keys.forall(_.containingMessage eq scalaDescriptor), "FieldDescriptor does not match message type.")
      snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig(
        inferencerConfig = __fieldsMap.get(scalaDescriptor.findFieldByNumber(1).get).flatMap(_.as[_root_.scala.Option[snapchat.research.gbml.gigl_resource_config.VertexAiResourceConfig]]).map(snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.InferencerConfig.VertexAiInferencerConfig(_))
            .orElse[snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.InferencerConfig](__fieldsMap.get(scalaDescriptor.findFieldByNumber(2).get).flatMap(_.as[_root_.scala.Option[snapchat.research.gbml.gigl_resource_config.DataflowResourceConfig]]).map(snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.InferencerConfig.DataflowInferencerConfig(_)))
            .orElse[snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.InferencerConfig](__fieldsMap.get(scalaDescriptor.findFieldByNumber(3).get).flatMap(_.as[_root_.scala.Option[snapchat.research.gbml.gigl_resource_config.LocalResourceConfig]]).map(snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.InferencerConfig.LocalInferencerConfig(_)))
            .getOrElse(snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.InferencerConfig.Empty)
      )
    case _ => throw new RuntimeException("Expected PMessage")
  }
  def javaDescriptor: _root_.com.google.protobuf.Descriptors.Descriptor = GiglResourceConfigProto.javaDescriptor.getMessageTypes().get(11)
  def scalaDescriptor: _root_.scalapb.descriptors.Descriptor = GiglResourceConfigProto.scalaDescriptor.messages(11)
  def messageCompanionForFieldNumber(__number: _root_.scala.Int): _root_.scalapb.GeneratedMessageCompanion[_] = {
    var __out: _root_.scalapb.GeneratedMessageCompanion[_] = null
    (__number: @_root_.scala.unchecked) match {
      case 1 => __out = snapchat.research.gbml.gigl_resource_config.VertexAiResourceConfig
      case 2 => __out = snapchat.research.gbml.gigl_resource_config.DataflowResourceConfig
      case 3 => __out = snapchat.research.gbml.gigl_resource_config.LocalResourceConfig
    }
    __out
  }
  lazy val nestedMessagesCompanions: Seq[_root_.scalapb.GeneratedMessageCompanion[_ <: _root_.scalapb.GeneratedMessage]] = Seq.empty
  def enumCompanionForFieldNumber(__fieldNumber: _root_.scala.Int): _root_.scalapb.GeneratedEnumCompanion[_] = throw new MatchError(__fieldNumber)
  lazy val defaultInstance = snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig(
    inferencerConfig = snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.InferencerConfig.Empty
  )
  sealed trait InferencerConfig extends _root_.scalapb.GeneratedOneof {
    def isEmpty: _root_.scala.Boolean = false
    def isDefined: _root_.scala.Boolean = true
    def isVertexAiInferencerConfig: _root_.scala.Boolean = false
    def isDataflowInferencerConfig: _root_.scala.Boolean = false
    def isLocalInferencerConfig: _root_.scala.Boolean = false
    def vertexAiInferencerConfig: _root_.scala.Option[snapchat.research.gbml.gigl_resource_config.VertexAiResourceConfig] = _root_.scala.None
    def dataflowInferencerConfig: _root_.scala.Option[snapchat.research.gbml.gigl_resource_config.DataflowResourceConfig] = _root_.scala.None
    def localInferencerConfig: _root_.scala.Option[snapchat.research.gbml.gigl_resource_config.LocalResourceConfig] = _root_.scala.None
  }
  object InferencerConfig {
    @SerialVersionUID(0L)
    case object Empty extends snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.InferencerConfig {
      type ValueType = _root_.scala.Nothing
      override def isEmpty: _root_.scala.Boolean = true
      override def isDefined: _root_.scala.Boolean = false
      override def number: _root_.scala.Int = 0
      override def value: _root_.scala.Nothing = throw new java.util.NoSuchElementException("Empty.value")
    }
  
    @SerialVersionUID(0L)
    final case class VertexAiInferencerConfig(value: snapchat.research.gbml.gigl_resource_config.VertexAiResourceConfig) extends snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.InferencerConfig {
      type ValueType = snapchat.research.gbml.gigl_resource_config.VertexAiResourceConfig
      override def isVertexAiInferencerConfig: _root_.scala.Boolean = true
      override def vertexAiInferencerConfig: _root_.scala.Option[snapchat.research.gbml.gigl_resource_config.VertexAiResourceConfig] = Some(value)
      override def number: _root_.scala.Int = 1
    }
    @SerialVersionUID(0L)
    final case class DataflowInferencerConfig(value: snapchat.research.gbml.gigl_resource_config.DataflowResourceConfig) extends snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.InferencerConfig {
      type ValueType = snapchat.research.gbml.gigl_resource_config.DataflowResourceConfig
      override def isDataflowInferencerConfig: _root_.scala.Boolean = true
      override def dataflowInferencerConfig: _root_.scala.Option[snapchat.research.gbml.gigl_resource_config.DataflowResourceConfig] = Some(value)
      override def number: _root_.scala.Int = 2
    }
    @SerialVersionUID(0L)
    final case class LocalInferencerConfig(value: snapchat.research.gbml.gigl_resource_config.LocalResourceConfig) extends snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.InferencerConfig {
      type ValueType = snapchat.research.gbml.gigl_resource_config.LocalResourceConfig
      override def isLocalInferencerConfig: _root_.scala.Boolean = true
      override def localInferencerConfig: _root_.scala.Option[snapchat.research.gbml.gigl_resource_config.LocalResourceConfig] = Some(value)
      override def number: _root_.scala.Int = 3
    }
  }
  implicit class InferencerResourceConfigLens[UpperPB](_l: _root_.scalapb.lenses.Lens[UpperPB, snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig]) extends _root_.scalapb.lenses.ObjectLens[UpperPB, snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig](_l) {
    def vertexAiInferencerConfig: _root_.scalapb.lenses.Lens[UpperPB, snapchat.research.gbml.gigl_resource_config.VertexAiResourceConfig] = field(_.getVertexAiInferencerConfig)((c_, f_) => c_.copy(inferencerConfig = snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.InferencerConfig.VertexAiInferencerConfig(f_)))
    def dataflowInferencerConfig: _root_.scalapb.lenses.Lens[UpperPB, snapchat.research.gbml.gigl_resource_config.DataflowResourceConfig] = field(_.getDataflowInferencerConfig)((c_, f_) => c_.copy(inferencerConfig = snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.InferencerConfig.DataflowInferencerConfig(f_)))
    def localInferencerConfig: _root_.scalapb.lenses.Lens[UpperPB, snapchat.research.gbml.gigl_resource_config.LocalResourceConfig] = field(_.getLocalInferencerConfig)((c_, f_) => c_.copy(inferencerConfig = snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.InferencerConfig.LocalInferencerConfig(f_)))
    def inferencerConfig: _root_.scalapb.lenses.Lens[UpperPB, snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.InferencerConfig] = field(_.inferencerConfig)((c_, f_) => c_.copy(inferencerConfig = f_))
  }
  final val VERTEX_AI_INFERENCER_CONFIG_FIELD_NUMBER = 1
  final val DATAFLOW_INFERENCER_CONFIG_FIELD_NUMBER = 2
  final val LOCAL_INFERENCER_CONFIG_FIELD_NUMBER = 3
  def of(
    inferencerConfig: snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig.InferencerConfig
  ): _root_.snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig = _root_.snapchat.research.gbml.gigl_resource_config.InferencerResourceConfig(
    inferencerConfig
  )
  // @@protoc_insertion_point(GeneratedMessageCompanion[snapchat.research.gbml.InferencerResourceConfig])
}
