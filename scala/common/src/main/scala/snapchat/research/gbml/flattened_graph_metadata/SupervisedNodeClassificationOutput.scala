// Generated by the Scala Plugin for the Protocol Buffer Compiler.
// Do not edit!
//
// Protofile syntax: PROTO3

package snapchat.research.gbml.flattened_graph_metadata

/** Stores SupervisedNodeClassificationSample-relevant output
  *
  * @param labeledTfrecordUriPrefix
  *   GCS prefix which can be used to glob the TFRecord dataset.
  */
@SerialVersionUID(0L)
final case class SupervisedNodeClassificationOutput(
    labeledTfrecordUriPrefix: _root_.scala.Predef.String = "",
    unlabeledTfrecordUriPrefix: _root_.scala.Predef.String = "",
    unknownFields: _root_.scalapb.UnknownFieldSet = _root_.scalapb.UnknownFieldSet.empty
    ) extends scalapb.GeneratedMessage with scalapb.lenses.Updatable[SupervisedNodeClassificationOutput] {
    @transient
    private[this] var __serializedSizeMemoized: _root_.scala.Int = 0
    private[this] def __computeSerializedSize(): _root_.scala.Int = {
      var __size = 0
      
      {
        val __value = labeledTfrecordUriPrefix
        if (!__value.isEmpty) {
          __size += _root_.com.google.protobuf.CodedOutputStream.computeStringSize(1, __value)
        }
      };
      
      {
        val __value = unlabeledTfrecordUriPrefix
        if (!__value.isEmpty) {
          __size += _root_.com.google.protobuf.CodedOutputStream.computeStringSize(2, __value)
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
        val __v = labeledTfrecordUriPrefix
        if (!__v.isEmpty) {
          _output__.writeString(1, __v)
        }
      };
      {
        val __v = unlabeledTfrecordUriPrefix
        if (!__v.isEmpty) {
          _output__.writeString(2, __v)
        }
      };
      unknownFields.writeTo(_output__)
    }
    def withLabeledTfrecordUriPrefix(__v: _root_.scala.Predef.String): SupervisedNodeClassificationOutput = copy(labeledTfrecordUriPrefix = __v)
    def withUnlabeledTfrecordUriPrefix(__v: _root_.scala.Predef.String): SupervisedNodeClassificationOutput = copy(unlabeledTfrecordUriPrefix = __v)
    def withUnknownFields(__v: _root_.scalapb.UnknownFieldSet) = copy(unknownFields = __v)
    def discardUnknownFields = copy(unknownFields = _root_.scalapb.UnknownFieldSet.empty)
    def getFieldByNumber(__fieldNumber: _root_.scala.Int): _root_.scala.Any = {
      (__fieldNumber: @_root_.scala.unchecked) match {
        case 1 => {
          val __t = labeledTfrecordUriPrefix
          if (__t != "") __t else null
        }
        case 2 => {
          val __t = unlabeledTfrecordUriPrefix
          if (__t != "") __t else null
        }
      }
    }
    def getField(__field: _root_.scalapb.descriptors.FieldDescriptor): _root_.scalapb.descriptors.PValue = {
      _root_.scala.Predef.require(__field.containingMessage eq companion.scalaDescriptor)
      (__field.number: @_root_.scala.unchecked) match {
        case 1 => _root_.scalapb.descriptors.PString(labeledTfrecordUriPrefix)
        case 2 => _root_.scalapb.descriptors.PString(unlabeledTfrecordUriPrefix)
      }
    }
    def toProtoString: _root_.scala.Predef.String = _root_.scalapb.TextFormat.printToUnicodeString(this)
    def companion: snapchat.research.gbml.flattened_graph_metadata.SupervisedNodeClassificationOutput.type = snapchat.research.gbml.flattened_graph_metadata.SupervisedNodeClassificationOutput
    // @@protoc_insertion_point(GeneratedMessage[snapchat.research.gbml.SupervisedNodeClassificationOutput])
}

object SupervisedNodeClassificationOutput extends scalapb.GeneratedMessageCompanion[snapchat.research.gbml.flattened_graph_metadata.SupervisedNodeClassificationOutput] {
  implicit def messageCompanion: scalapb.GeneratedMessageCompanion[snapchat.research.gbml.flattened_graph_metadata.SupervisedNodeClassificationOutput] = this
  def parseFrom(`_input__`: _root_.com.google.protobuf.CodedInputStream): snapchat.research.gbml.flattened_graph_metadata.SupervisedNodeClassificationOutput = {
    var __labeledTfrecordUriPrefix: _root_.scala.Predef.String = ""
    var __unlabeledTfrecordUriPrefix: _root_.scala.Predef.String = ""
    var `_unknownFields__`: _root_.scalapb.UnknownFieldSet.Builder = null
    var _done__ = false
    while (!_done__) {
      val _tag__ = _input__.readTag()
      _tag__ match {
        case 0 => _done__ = true
        case 10 =>
          __labeledTfrecordUriPrefix = _input__.readStringRequireUtf8()
        case 18 =>
          __unlabeledTfrecordUriPrefix = _input__.readStringRequireUtf8()
        case tag =>
          if (_unknownFields__ == null) {
            _unknownFields__ = new _root_.scalapb.UnknownFieldSet.Builder()
          }
          _unknownFields__.parseField(tag, _input__)
      }
    }
    snapchat.research.gbml.flattened_graph_metadata.SupervisedNodeClassificationOutput(
        labeledTfrecordUriPrefix = __labeledTfrecordUriPrefix,
        unlabeledTfrecordUriPrefix = __unlabeledTfrecordUriPrefix,
        unknownFields = if (_unknownFields__ == null) _root_.scalapb.UnknownFieldSet.empty else _unknownFields__.result()
    )
  }
  implicit def messageReads: _root_.scalapb.descriptors.Reads[snapchat.research.gbml.flattened_graph_metadata.SupervisedNodeClassificationOutput] = _root_.scalapb.descriptors.Reads{
    case _root_.scalapb.descriptors.PMessage(__fieldsMap) =>
      _root_.scala.Predef.require(__fieldsMap.keys.forall(_.containingMessage eq scalaDescriptor), "FieldDescriptor does not match message type.")
      snapchat.research.gbml.flattened_graph_metadata.SupervisedNodeClassificationOutput(
        labeledTfrecordUriPrefix = __fieldsMap.get(scalaDescriptor.findFieldByNumber(1).get).map(_.as[_root_.scala.Predef.String]).getOrElse(""),
        unlabeledTfrecordUriPrefix = __fieldsMap.get(scalaDescriptor.findFieldByNumber(2).get).map(_.as[_root_.scala.Predef.String]).getOrElse("")
      )
    case _ => throw new RuntimeException("Expected PMessage")
  }
  def javaDescriptor: _root_.com.google.protobuf.Descriptors.Descriptor = FlattenedGraphMetadataProto.javaDescriptor.getMessageTypes().get(0)
  def scalaDescriptor: _root_.scalapb.descriptors.Descriptor = FlattenedGraphMetadataProto.scalaDescriptor.messages(0)
  def messageCompanionForFieldNumber(__number: _root_.scala.Int): _root_.scalapb.GeneratedMessageCompanion[_] = throw new MatchError(__number)
  lazy val nestedMessagesCompanions: Seq[_root_.scalapb.GeneratedMessageCompanion[_ <: _root_.scalapb.GeneratedMessage]] = Seq.empty
  def enumCompanionForFieldNumber(__fieldNumber: _root_.scala.Int): _root_.scalapb.GeneratedEnumCompanion[_] = throw new MatchError(__fieldNumber)
  lazy val defaultInstance = snapchat.research.gbml.flattened_graph_metadata.SupervisedNodeClassificationOutput(
    labeledTfrecordUriPrefix = "",
    unlabeledTfrecordUriPrefix = ""
  )
  implicit class SupervisedNodeClassificationOutputLens[UpperPB](_l: _root_.scalapb.lenses.Lens[UpperPB, snapchat.research.gbml.flattened_graph_metadata.SupervisedNodeClassificationOutput]) extends _root_.scalapb.lenses.ObjectLens[UpperPB, snapchat.research.gbml.flattened_graph_metadata.SupervisedNodeClassificationOutput](_l) {
    def labeledTfrecordUriPrefix: _root_.scalapb.lenses.Lens[UpperPB, _root_.scala.Predef.String] = field(_.labeledTfrecordUriPrefix)((c_, f_) => c_.copy(labeledTfrecordUriPrefix = f_))
    def unlabeledTfrecordUriPrefix: _root_.scalapb.lenses.Lens[UpperPB, _root_.scala.Predef.String] = field(_.unlabeledTfrecordUriPrefix)((c_, f_) => c_.copy(unlabeledTfrecordUriPrefix = f_))
  }
  final val LABELED_TFRECORD_URI_PREFIX_FIELD_NUMBER = 1
  final val UNLABELED_TFRECORD_URI_PREFIX_FIELD_NUMBER = 2
  def of(
    labeledTfrecordUriPrefix: _root_.scala.Predef.String,
    unlabeledTfrecordUriPrefix: _root_.scala.Predef.String
  ): _root_.snapchat.research.gbml.flattened_graph_metadata.SupervisedNodeClassificationOutput = _root_.snapchat.research.gbml.flattened_graph_metadata.SupervisedNodeClassificationOutput(
    labeledTfrecordUriPrefix,
    unlabeledTfrecordUriPrefix
  )
  // @@protoc_insertion_point(GeneratedMessageCompanion[snapchat.research.gbml.SupervisedNodeClassificationOutput])
}
