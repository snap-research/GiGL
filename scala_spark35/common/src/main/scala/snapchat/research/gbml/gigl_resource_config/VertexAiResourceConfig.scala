// Generated by the Scala Plugin for the Protocol Buffer Compiler.
// Do not edit!
//
// Protofile syntax: PROTO3

package snapchat.research.gbml.gigl_resource_config

/** Configuration for Vertex AI resources
  *
  * @param machineType
  *   Machine type for job
  * @param gpuType
  *   GPU type for job. Must be set to 'ACCELERATOR_TYPE_UNSPECIFIED' for cpu.
  * @param gpuLimit
  *   GPU limit for job. Must be set to 0 for cpu.
  * @param numReplicas
  *   Num workers for job
  * @param timeout
  *   Timeout in seconds for the job. If unset or zero, will use the default &#64; google.cloud.aiplatform.CustomJob, which is 7 days: 
  *   https://github.com/googleapis/python-aiplatform/blob/58fbabdeeefd1ccf1a9d0c22eeb5606aeb9c2266/google/cloud/aiplatform/jobs.py#L2252-L2253
  */
@SerialVersionUID(0L)
final case class VertexAiResourceConfig(
    machineType: _root_.scala.Predef.String = "",
    gpuType: _root_.scala.Predef.String = "",
    gpuLimit: _root_.scala.Int = 0,
    numReplicas: _root_.scala.Int = 0,
    timeout: _root_.scala.Int = 0,
    unknownFields: _root_.scalapb.UnknownFieldSet = _root_.scalapb.UnknownFieldSet.empty
    ) extends scalapb.GeneratedMessage with scalapb.lenses.Updatable[VertexAiResourceConfig] {
    @transient
    private[this] var __serializedSizeMemoized: _root_.scala.Int = 0
    private[this] def __computeSerializedSize(): _root_.scala.Int = {
      var __size = 0
      
      {
        val __value = machineType
        if (!__value.isEmpty) {
          __size += _root_.com.google.protobuf.CodedOutputStream.computeStringSize(1, __value)
        }
      };
      
      {
        val __value = gpuType
        if (!__value.isEmpty) {
          __size += _root_.com.google.protobuf.CodedOutputStream.computeStringSize(2, __value)
        }
      };
      
      {
        val __value = gpuLimit
        if (__value != 0) {
          __size += _root_.com.google.protobuf.CodedOutputStream.computeUInt32Size(3, __value)
        }
      };
      
      {
        val __value = numReplicas
        if (__value != 0) {
          __size += _root_.com.google.protobuf.CodedOutputStream.computeUInt32Size(4, __value)
        }
      };
      
      {
        val __value = timeout
        if (__value != 0) {
          __size += _root_.com.google.protobuf.CodedOutputStream.computeUInt32Size(5, __value)
        }
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
      {
        val __v = machineType
        if (!__v.isEmpty) {
          _output__.writeString(1, __v)
        }
      };
      {
        val __v = gpuType
        if (!__v.isEmpty) {
          _output__.writeString(2, __v)
        }
      };
      {
        val __v = gpuLimit
        if (__v != 0) {
          _output__.writeUInt32(3, __v)
        }
      };
      {
        val __v = numReplicas
        if (__v != 0) {
          _output__.writeUInt32(4, __v)
        }
      };
      {
        val __v = timeout
        if (__v != 0) {
          _output__.writeUInt32(5, __v)
        }
      };
      unknownFields.writeTo(_output__)
    }
    def withMachineType(__v: _root_.scala.Predef.String): VertexAiResourceConfig = copy(machineType = __v)
    def withGpuType(__v: _root_.scala.Predef.String): VertexAiResourceConfig = copy(gpuType = __v)
    def withGpuLimit(__v: _root_.scala.Int): VertexAiResourceConfig = copy(gpuLimit = __v)
    def withNumReplicas(__v: _root_.scala.Int): VertexAiResourceConfig = copy(numReplicas = __v)
    def withTimeout(__v: _root_.scala.Int): VertexAiResourceConfig = copy(timeout = __v)
    def withUnknownFields(__v: _root_.scalapb.UnknownFieldSet) = copy(unknownFields = __v)
    def discardUnknownFields = copy(unknownFields = _root_.scalapb.UnknownFieldSet.empty)
    def getFieldByNumber(__fieldNumber: _root_.scala.Int): _root_.scala.Any = {
      (__fieldNumber: @_root_.scala.unchecked) match {
        case 1 => {
          val __t = machineType
          if (__t != "") __t else null
        }
        case 2 => {
          val __t = gpuType
          if (__t != "") __t else null
        }
        case 3 => {
          val __t = gpuLimit
          if (__t != 0) __t else null
        }
        case 4 => {
          val __t = numReplicas
          if (__t != 0) __t else null
        }
        case 5 => {
          val __t = timeout
          if (__t != 0) __t else null
        }
      }
    }
    def getField(__field: _root_.scalapb.descriptors.FieldDescriptor): _root_.scalapb.descriptors.PValue = {
      _root_.scala.Predef.require(__field.containingMessage eq companion.scalaDescriptor)
      (__field.number: @_root_.scala.unchecked) match {
        case 1 => _root_.scalapb.descriptors.PString(machineType)
        case 2 => _root_.scalapb.descriptors.PString(gpuType)
        case 3 => _root_.scalapb.descriptors.PInt(gpuLimit)
        case 4 => _root_.scalapb.descriptors.PInt(numReplicas)
        case 5 => _root_.scalapb.descriptors.PInt(timeout)
      }
    }
    def toProtoString: _root_.scala.Predef.String = _root_.scalapb.TextFormat.printToUnicodeString(this)
    def companion: snapchat.research.gbml.gigl_resource_config.VertexAiResourceConfig.type = snapchat.research.gbml.gigl_resource_config.VertexAiResourceConfig
    // @@protoc_insertion_point(GeneratedMessage[snapchat.research.gbml.VertexAiResourceConfig])
}

object VertexAiResourceConfig extends scalapb.GeneratedMessageCompanion[snapchat.research.gbml.gigl_resource_config.VertexAiResourceConfig] {
  implicit def messageCompanion: scalapb.GeneratedMessageCompanion[snapchat.research.gbml.gigl_resource_config.VertexAiResourceConfig] = this
  def parseFrom(`_input__`: _root_.com.google.protobuf.CodedInputStream): snapchat.research.gbml.gigl_resource_config.VertexAiResourceConfig = {
    var __machineType: _root_.scala.Predef.String = ""
    var __gpuType: _root_.scala.Predef.String = ""
    var __gpuLimit: _root_.scala.Int = 0
    var __numReplicas: _root_.scala.Int = 0
    var __timeout: _root_.scala.Int = 0
    var `_unknownFields__`: _root_.scalapb.UnknownFieldSet.Builder = null
    var _done__ = false
    while (!_done__) {
      val _tag__ = _input__.readTag()
      _tag__ match {
        case 0 => _done__ = true
        case 10 =>
          __machineType = _input__.readStringRequireUtf8()
        case 18 =>
          __gpuType = _input__.readStringRequireUtf8()
        case 24 =>
          __gpuLimit = _input__.readUInt32()
        case 32 =>
          __numReplicas = _input__.readUInt32()
        case 40 =>
          __timeout = _input__.readUInt32()
        case tag =>
          if (_unknownFields__ == null) {
            _unknownFields__ = new _root_.scalapb.UnknownFieldSet.Builder()
          }
          _unknownFields__.parseField(tag, _input__)
      }
    }
    snapchat.research.gbml.gigl_resource_config.VertexAiResourceConfig(
        machineType = __machineType,
        gpuType = __gpuType,
        gpuLimit = __gpuLimit,
        numReplicas = __numReplicas,
        timeout = __timeout,
        unknownFields = if (_unknownFields__ == null) _root_.scalapb.UnknownFieldSet.empty else _unknownFields__.result()
    )
  }
  implicit def messageReads: _root_.scalapb.descriptors.Reads[snapchat.research.gbml.gigl_resource_config.VertexAiResourceConfig] = _root_.scalapb.descriptors.Reads{
    case _root_.scalapb.descriptors.PMessage(__fieldsMap) =>
      _root_.scala.Predef.require(__fieldsMap.keys.forall(_.containingMessage eq scalaDescriptor), "FieldDescriptor does not match message type.")
      snapchat.research.gbml.gigl_resource_config.VertexAiResourceConfig(
        machineType = __fieldsMap.get(scalaDescriptor.findFieldByNumber(1).get).map(_.as[_root_.scala.Predef.String]).getOrElse(""),
        gpuType = __fieldsMap.get(scalaDescriptor.findFieldByNumber(2).get).map(_.as[_root_.scala.Predef.String]).getOrElse(""),
        gpuLimit = __fieldsMap.get(scalaDescriptor.findFieldByNumber(3).get).map(_.as[_root_.scala.Int]).getOrElse(0),
        numReplicas = __fieldsMap.get(scalaDescriptor.findFieldByNumber(4).get).map(_.as[_root_.scala.Int]).getOrElse(0),
        timeout = __fieldsMap.get(scalaDescriptor.findFieldByNumber(5).get).map(_.as[_root_.scala.Int]).getOrElse(0)
      )
    case _ => throw new RuntimeException("Expected PMessage")
  }
  def javaDescriptor: _root_.com.google.protobuf.Descriptors.Descriptor = GiglResourceConfigProto.javaDescriptor.getMessageTypes().get(6)
  def scalaDescriptor: _root_.scalapb.descriptors.Descriptor = GiglResourceConfigProto.scalaDescriptor.messages(6)
  def messageCompanionForFieldNumber(__number: _root_.scala.Int): _root_.scalapb.GeneratedMessageCompanion[_] = throw new MatchError(__number)
  lazy val nestedMessagesCompanions: Seq[_root_.scalapb.GeneratedMessageCompanion[_ <: _root_.scalapb.GeneratedMessage]] = Seq.empty
  def enumCompanionForFieldNumber(__fieldNumber: _root_.scala.Int): _root_.scalapb.GeneratedEnumCompanion[_] = throw new MatchError(__fieldNumber)
  lazy val defaultInstance = snapchat.research.gbml.gigl_resource_config.VertexAiResourceConfig(
    machineType = "",
    gpuType = "",
    gpuLimit = 0,
    numReplicas = 0,
    timeout = 0
  )
  implicit class VertexAiResourceConfigLens[UpperPB](_l: _root_.scalapb.lenses.Lens[UpperPB, snapchat.research.gbml.gigl_resource_config.VertexAiResourceConfig]) extends _root_.scalapb.lenses.ObjectLens[UpperPB, snapchat.research.gbml.gigl_resource_config.VertexAiResourceConfig](_l) {
    def machineType: _root_.scalapb.lenses.Lens[UpperPB, _root_.scala.Predef.String] = field(_.machineType)((c_, f_) => c_.copy(machineType = f_))
    def gpuType: _root_.scalapb.lenses.Lens[UpperPB, _root_.scala.Predef.String] = field(_.gpuType)((c_, f_) => c_.copy(gpuType = f_))
    def gpuLimit: _root_.scalapb.lenses.Lens[UpperPB, _root_.scala.Int] = field(_.gpuLimit)((c_, f_) => c_.copy(gpuLimit = f_))
    def numReplicas: _root_.scalapb.lenses.Lens[UpperPB, _root_.scala.Int] = field(_.numReplicas)((c_, f_) => c_.copy(numReplicas = f_))
    def timeout: _root_.scalapb.lenses.Lens[UpperPB, _root_.scala.Int] = field(_.timeout)((c_, f_) => c_.copy(timeout = f_))
  }
  final val MACHINE_TYPE_FIELD_NUMBER = 1
  final val GPU_TYPE_FIELD_NUMBER = 2
  final val GPU_LIMIT_FIELD_NUMBER = 3
  final val NUM_REPLICAS_FIELD_NUMBER = 4
  final val TIMEOUT_FIELD_NUMBER = 5
  def of(
    machineType: _root_.scala.Predef.String,
    gpuType: _root_.scala.Predef.String,
    gpuLimit: _root_.scala.Int,
    numReplicas: _root_.scala.Int,
    timeout: _root_.scala.Int
  ): _root_.snapchat.research.gbml.gigl_resource_config.VertexAiResourceConfig = _root_.snapchat.research.gbml.gigl_resource_config.VertexAiResourceConfig(
    machineType,
    gpuType,
    gpuLimit,
    numReplicas,
    timeout
  )
  // @@protoc_insertion_point(GeneratedMessageCompanion[snapchat.research.gbml.VertexAiResourceConfig])
}
