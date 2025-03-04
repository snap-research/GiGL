// Generated by the Scala Plugin for the Protocol Buffer Compiler.
// Do not edit!
//
// Protofile syntax: PROTO3

package snapchat.research.gbml.gigl_resource_config

/** (deprecated)
  * Configuration for distributed training resources
  */
@SerialVersionUID(0L)
final case class DistributedTrainerConfig(
    trainerConfig: snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.TrainerConfig = snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.TrainerConfig.Empty,
    unknownFields: _root_.scalapb.UnknownFieldSet = _root_.scalapb.UnknownFieldSet.empty
    ) extends scalapb.GeneratedMessage with scalapb.lenses.Updatable[DistributedTrainerConfig] {
    @transient
    private[this] var __serializedSizeMemoized: _root_.scala.Int = 0
    private[this] def __computeSerializedSize(): _root_.scala.Int = {
      var __size = 0
      if (trainerConfig.vertexAiTrainerConfig.isDefined) {
        val __value = trainerConfig.vertexAiTrainerConfig.get
        __size += 1 + _root_.com.google.protobuf.CodedOutputStream.computeUInt32SizeNoTag(__value.serializedSize) + __value.serializedSize
      };
      if (trainerConfig.kfpTrainerConfig.isDefined) {
        val __value = trainerConfig.kfpTrainerConfig.get
        __size += 1 + _root_.com.google.protobuf.CodedOutputStream.computeUInt32SizeNoTag(__value.serializedSize) + __value.serializedSize
      };
      if (trainerConfig.localTrainerConfig.isDefined) {
        val __value = trainerConfig.localTrainerConfig.get
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
      trainerConfig.vertexAiTrainerConfig.foreach { __v =>
        val __m = __v
        _output__.writeTag(1, 2)
        _output__.writeUInt32NoTag(__m.serializedSize)
        __m.writeTo(_output__)
      };
      trainerConfig.kfpTrainerConfig.foreach { __v =>
        val __m = __v
        _output__.writeTag(2, 2)
        _output__.writeUInt32NoTag(__m.serializedSize)
        __m.writeTo(_output__)
      };
      trainerConfig.localTrainerConfig.foreach { __v =>
        val __m = __v
        _output__.writeTag(3, 2)
        _output__.writeUInt32NoTag(__m.serializedSize)
        __m.writeTo(_output__)
      };
      unknownFields.writeTo(_output__)
    }
    def getVertexAiTrainerConfig: snapchat.research.gbml.gigl_resource_config.VertexAiTrainerConfig = trainerConfig.vertexAiTrainerConfig.getOrElse(snapchat.research.gbml.gigl_resource_config.VertexAiTrainerConfig.defaultInstance)
    def withVertexAiTrainerConfig(__v: snapchat.research.gbml.gigl_resource_config.VertexAiTrainerConfig): DistributedTrainerConfig = copy(trainerConfig = snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.TrainerConfig.VertexAiTrainerConfig(__v))
    def getKfpTrainerConfig: snapchat.research.gbml.gigl_resource_config.KFPTrainerConfig = trainerConfig.kfpTrainerConfig.getOrElse(snapchat.research.gbml.gigl_resource_config.KFPTrainerConfig.defaultInstance)
    def withKfpTrainerConfig(__v: snapchat.research.gbml.gigl_resource_config.KFPTrainerConfig): DistributedTrainerConfig = copy(trainerConfig = snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.TrainerConfig.KfpTrainerConfig(__v))
    def getLocalTrainerConfig: snapchat.research.gbml.gigl_resource_config.LocalTrainerConfig = trainerConfig.localTrainerConfig.getOrElse(snapchat.research.gbml.gigl_resource_config.LocalTrainerConfig.defaultInstance)
    def withLocalTrainerConfig(__v: snapchat.research.gbml.gigl_resource_config.LocalTrainerConfig): DistributedTrainerConfig = copy(trainerConfig = snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.TrainerConfig.LocalTrainerConfig(__v))
    def clearTrainerConfig: DistributedTrainerConfig = copy(trainerConfig = snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.TrainerConfig.Empty)
    def withTrainerConfig(__v: snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.TrainerConfig): DistributedTrainerConfig = copy(trainerConfig = __v)
    def withUnknownFields(__v: _root_.scalapb.UnknownFieldSet) = copy(unknownFields = __v)
    def discardUnknownFields = copy(unknownFields = _root_.scalapb.UnknownFieldSet.empty)
    def getFieldByNumber(__fieldNumber: _root_.scala.Int): _root_.scala.Any = {
      (__fieldNumber: @_root_.scala.unchecked) match {
        case 1 => trainerConfig.vertexAiTrainerConfig.orNull
        case 2 => trainerConfig.kfpTrainerConfig.orNull
        case 3 => trainerConfig.localTrainerConfig.orNull
      }
    }
    def getField(__field: _root_.scalapb.descriptors.FieldDescriptor): _root_.scalapb.descriptors.PValue = {
      _root_.scala.Predef.require(__field.containingMessage eq companion.scalaDescriptor)
      (__field.number: @_root_.scala.unchecked) match {
        case 1 => trainerConfig.vertexAiTrainerConfig.map(_.toPMessage).getOrElse(_root_.scalapb.descriptors.PEmpty)
        case 2 => trainerConfig.kfpTrainerConfig.map(_.toPMessage).getOrElse(_root_.scalapb.descriptors.PEmpty)
        case 3 => trainerConfig.localTrainerConfig.map(_.toPMessage).getOrElse(_root_.scalapb.descriptors.PEmpty)
      }
    }
    def toProtoString: _root_.scala.Predef.String = _root_.scalapb.TextFormat.printToUnicodeString(this)
    def companion: snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.type = snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig
    // @@protoc_insertion_point(GeneratedMessage[snapchat.research.gbml.DistributedTrainerConfig])
}

object DistributedTrainerConfig extends scalapb.GeneratedMessageCompanion[snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig] {
  implicit def messageCompanion: scalapb.GeneratedMessageCompanion[snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig] = this
  def parseFrom(`_input__`: _root_.com.google.protobuf.CodedInputStream): snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig = {
    var __trainerConfig: snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.TrainerConfig = snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.TrainerConfig.Empty
    var `_unknownFields__`: _root_.scalapb.UnknownFieldSet.Builder = null
    var _done__ = false
    while (!_done__) {
      val _tag__ = _input__.readTag()
      _tag__ match {
        case 0 => _done__ = true
        case 10 =>
          __trainerConfig = snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.TrainerConfig.VertexAiTrainerConfig(__trainerConfig.vertexAiTrainerConfig.fold(_root_.scalapb.LiteParser.readMessage[snapchat.research.gbml.gigl_resource_config.VertexAiTrainerConfig](_input__))(_root_.scalapb.LiteParser.readMessage(_input__, _)))
        case 18 =>
          __trainerConfig = snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.TrainerConfig.KfpTrainerConfig(__trainerConfig.kfpTrainerConfig.fold(_root_.scalapb.LiteParser.readMessage[snapchat.research.gbml.gigl_resource_config.KFPTrainerConfig](_input__))(_root_.scalapb.LiteParser.readMessage(_input__, _)))
        case 26 =>
          __trainerConfig = snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.TrainerConfig.LocalTrainerConfig(__trainerConfig.localTrainerConfig.fold(_root_.scalapb.LiteParser.readMessage[snapchat.research.gbml.gigl_resource_config.LocalTrainerConfig](_input__))(_root_.scalapb.LiteParser.readMessage(_input__, _)))
        case tag =>
          if (_unknownFields__ == null) {
            _unknownFields__ = new _root_.scalapb.UnknownFieldSet.Builder()
          }
          _unknownFields__.parseField(tag, _input__)
      }
    }
    snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig(
        trainerConfig = __trainerConfig,
        unknownFields = if (_unknownFields__ == null) _root_.scalapb.UnknownFieldSet.empty else _unknownFields__.result()
    )
  }
  implicit def messageReads: _root_.scalapb.descriptors.Reads[snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig] = _root_.scalapb.descriptors.Reads{
    case _root_.scalapb.descriptors.PMessage(__fieldsMap) =>
      _root_.scala.Predef.require(__fieldsMap.keys.forall(_.containingMessage eq scalaDescriptor), "FieldDescriptor does not match message type.")
      snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig(
        trainerConfig = __fieldsMap.get(scalaDescriptor.findFieldByNumber(1).get).flatMap(_.as[_root_.scala.Option[snapchat.research.gbml.gigl_resource_config.VertexAiTrainerConfig]]).map(snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.TrainerConfig.VertexAiTrainerConfig(_))
            .orElse[snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.TrainerConfig](__fieldsMap.get(scalaDescriptor.findFieldByNumber(2).get).flatMap(_.as[_root_.scala.Option[snapchat.research.gbml.gigl_resource_config.KFPTrainerConfig]]).map(snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.TrainerConfig.KfpTrainerConfig(_)))
            .orElse[snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.TrainerConfig](__fieldsMap.get(scalaDescriptor.findFieldByNumber(3).get).flatMap(_.as[_root_.scala.Option[snapchat.research.gbml.gigl_resource_config.LocalTrainerConfig]]).map(snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.TrainerConfig.LocalTrainerConfig(_)))
            .getOrElse(snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.TrainerConfig.Empty)
      )
    case _ => throw new RuntimeException("Expected PMessage")
  }
  def javaDescriptor: _root_.com.google.protobuf.Descriptors.Descriptor = GiglResourceConfigProto.javaDescriptor.getMessageTypes().get(9)
  def scalaDescriptor: _root_.scalapb.descriptors.Descriptor = GiglResourceConfigProto.scalaDescriptor.messages(9)
  def messageCompanionForFieldNumber(__number: _root_.scala.Int): _root_.scalapb.GeneratedMessageCompanion[_] = {
    var __out: _root_.scalapb.GeneratedMessageCompanion[_] = null
    (__number: @_root_.scala.unchecked) match {
      case 1 => __out = snapchat.research.gbml.gigl_resource_config.VertexAiTrainerConfig
      case 2 => __out = snapchat.research.gbml.gigl_resource_config.KFPTrainerConfig
      case 3 => __out = snapchat.research.gbml.gigl_resource_config.LocalTrainerConfig
    }
    __out
  }
  lazy val nestedMessagesCompanions: Seq[_root_.scalapb.GeneratedMessageCompanion[_ <: _root_.scalapb.GeneratedMessage]] = Seq.empty
  def enumCompanionForFieldNumber(__fieldNumber: _root_.scala.Int): _root_.scalapb.GeneratedEnumCompanion[_] = throw new MatchError(__fieldNumber)
  lazy val defaultInstance = snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig(
    trainerConfig = snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.TrainerConfig.Empty
  )
  sealed trait TrainerConfig extends _root_.scalapb.GeneratedOneof {
    def isEmpty: _root_.scala.Boolean = false
    def isDefined: _root_.scala.Boolean = true
    def isVertexAiTrainerConfig: _root_.scala.Boolean = false
    def isKfpTrainerConfig: _root_.scala.Boolean = false
    def isLocalTrainerConfig: _root_.scala.Boolean = false
    def vertexAiTrainerConfig: _root_.scala.Option[snapchat.research.gbml.gigl_resource_config.VertexAiTrainerConfig] = _root_.scala.None
    def kfpTrainerConfig: _root_.scala.Option[snapchat.research.gbml.gigl_resource_config.KFPTrainerConfig] = _root_.scala.None
    def localTrainerConfig: _root_.scala.Option[snapchat.research.gbml.gigl_resource_config.LocalTrainerConfig] = _root_.scala.None
  }
  object TrainerConfig {
    @SerialVersionUID(0L)
    case object Empty extends snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.TrainerConfig {
      type ValueType = _root_.scala.Nothing
      override def isEmpty: _root_.scala.Boolean = true
      override def isDefined: _root_.scala.Boolean = false
      override def number: _root_.scala.Int = 0
      override def value: _root_.scala.Nothing = throw new java.util.NoSuchElementException("Empty.value")
    }
  
    @SerialVersionUID(0L)
    final case class VertexAiTrainerConfig(value: snapchat.research.gbml.gigl_resource_config.VertexAiTrainerConfig) extends snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.TrainerConfig {
      type ValueType = snapchat.research.gbml.gigl_resource_config.VertexAiTrainerConfig
      override def isVertexAiTrainerConfig: _root_.scala.Boolean = true
      override def vertexAiTrainerConfig: _root_.scala.Option[snapchat.research.gbml.gigl_resource_config.VertexAiTrainerConfig] = Some(value)
      override def number: _root_.scala.Int = 1
    }
    @SerialVersionUID(0L)
    final case class KfpTrainerConfig(value: snapchat.research.gbml.gigl_resource_config.KFPTrainerConfig) extends snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.TrainerConfig {
      type ValueType = snapchat.research.gbml.gigl_resource_config.KFPTrainerConfig
      override def isKfpTrainerConfig: _root_.scala.Boolean = true
      override def kfpTrainerConfig: _root_.scala.Option[snapchat.research.gbml.gigl_resource_config.KFPTrainerConfig] = Some(value)
      override def number: _root_.scala.Int = 2
    }
    @SerialVersionUID(0L)
    final case class LocalTrainerConfig(value: snapchat.research.gbml.gigl_resource_config.LocalTrainerConfig) extends snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.TrainerConfig {
      type ValueType = snapchat.research.gbml.gigl_resource_config.LocalTrainerConfig
      override def isLocalTrainerConfig: _root_.scala.Boolean = true
      override def localTrainerConfig: _root_.scala.Option[snapchat.research.gbml.gigl_resource_config.LocalTrainerConfig] = Some(value)
      override def number: _root_.scala.Int = 3
    }
  }
  implicit class DistributedTrainerConfigLens[UpperPB](_l: _root_.scalapb.lenses.Lens[UpperPB, snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig]) extends _root_.scalapb.lenses.ObjectLens[UpperPB, snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig](_l) {
    def vertexAiTrainerConfig: _root_.scalapb.lenses.Lens[UpperPB, snapchat.research.gbml.gigl_resource_config.VertexAiTrainerConfig] = field(_.getVertexAiTrainerConfig)((c_, f_) => c_.copy(trainerConfig = snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.TrainerConfig.VertexAiTrainerConfig(f_)))
    def kfpTrainerConfig: _root_.scalapb.lenses.Lens[UpperPB, snapchat.research.gbml.gigl_resource_config.KFPTrainerConfig] = field(_.getKfpTrainerConfig)((c_, f_) => c_.copy(trainerConfig = snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.TrainerConfig.KfpTrainerConfig(f_)))
    def localTrainerConfig: _root_.scalapb.lenses.Lens[UpperPB, snapchat.research.gbml.gigl_resource_config.LocalTrainerConfig] = field(_.getLocalTrainerConfig)((c_, f_) => c_.copy(trainerConfig = snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.TrainerConfig.LocalTrainerConfig(f_)))
    def trainerConfig: _root_.scalapb.lenses.Lens[UpperPB, snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.TrainerConfig] = field(_.trainerConfig)((c_, f_) => c_.copy(trainerConfig = f_))
  }
  final val VERTEX_AI_TRAINER_CONFIG_FIELD_NUMBER = 1
  final val KFP_TRAINER_CONFIG_FIELD_NUMBER = 2
  final val LOCAL_TRAINER_CONFIG_FIELD_NUMBER = 3
  def of(
    trainerConfig: snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig.TrainerConfig
  ): _root_.snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig = _root_.snapchat.research.gbml.gigl_resource_config.DistributedTrainerConfig(
    trainerConfig
  )
  // @@protoc_insertion_point(GeneratedMessageCompanion[snapchat.research.gbml.DistributedTrainerConfig])
}
