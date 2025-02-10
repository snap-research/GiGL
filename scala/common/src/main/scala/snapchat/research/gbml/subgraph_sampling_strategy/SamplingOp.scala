// Generated by the Scala Plugin for the Protocol Buffer Compiler.
// Do not edit!
//
// Protofile syntax: PROTO3

package snapchat.research.gbml.subgraph_sampling_strategy

/** @param opName
  *   1-100 field numbers reserved for expanding SamplingOp
  *   Can be used as a reference in other operations
  * @param inputOpNames
  *   the list of upstream sampling operations in the DAG, empty if this is the root
  * @param samplingDirection
  *   Sampling edge direction, INCOMING or OUTGOING. INCOMING is default since SamplingDirection enum defined INCOMING=0, OUTGOING=1
  *   INCOMING - Sample incoming edges to the dst nodes (default)
  *   OUTGOING - Sample outgoing edges from the src nodes
  */
@SerialVersionUID(0L)
final case class SamplingOp(
    opName: _root_.scala.Predef.String = "",
    edgeType: _root_.scala.Option[snapchat.research.gbml.graph_schema.EdgeType] = _root_.scala.None,
    inputOpNames: _root_.scala.Seq[_root_.scala.Predef.String] = _root_.scala.Seq.empty,
    samplingMethod: snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod = snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod.Empty,
    samplingDirection: snapchat.research.gbml.subgraph_sampling_strategy.SamplingDirection = snapchat.research.gbml.subgraph_sampling_strategy.SamplingDirection.INCOMING,
    unknownFields: _root_.scalapb.UnknownFieldSet = _root_.scalapb.UnknownFieldSet.empty
    ) extends scalapb.GeneratedMessage with scalapb.lenses.Updatable[SamplingOp] {
    @transient
    private[this] var __serializedSizeMemoized: _root_.scala.Int = 0
    private[this] def __computeSerializedSize(): _root_.scala.Int = {
      var __size = 0
      
      {
        val __value = opName
        if (!__value.isEmpty) {
          __size += _root_.com.google.protobuf.CodedOutputStream.computeStringSize(1, __value)
        }
      };
      if (edgeType.isDefined) {
        val __value = edgeType.get
        __size += 1 + _root_.com.google.protobuf.CodedOutputStream.computeUInt32SizeNoTag(__value.serializedSize) + __value.serializedSize
      };
      inputOpNames.foreach { __item =>
        val __value = __item
        __size += _root_.com.google.protobuf.CodedOutputStream.computeStringSize(3, __value)
      }
      if (samplingMethod.randomUniform.isDefined) {
        val __value = samplingMethod.randomUniform.get
        __size += 2 + _root_.com.google.protobuf.CodedOutputStream.computeUInt32SizeNoTag(__value.serializedSize) + __value.serializedSize
      };
      if (samplingMethod.randomWeighted.isDefined) {
        val __value = samplingMethod.randomWeighted.get
        __size += 2 + _root_.com.google.protobuf.CodedOutputStream.computeUInt32SizeNoTag(__value.serializedSize) + __value.serializedSize
      };
      if (samplingMethod.topK.isDefined) {
        val __value = samplingMethod.topK.get
        __size += 2 + _root_.com.google.protobuf.CodedOutputStream.computeUInt32SizeNoTag(__value.serializedSize) + __value.serializedSize
      };
      if (samplingMethod.userDefined.isDefined) {
        val __value = samplingMethod.userDefined.get
        __size += 2 + _root_.com.google.protobuf.CodedOutputStream.computeUInt32SizeNoTag(__value.serializedSize) + __value.serializedSize
      };
      
      {
        val __value = samplingDirection.value
        if (__value != 0) {
          __size += _root_.com.google.protobuf.CodedOutputStream.computeEnumSize(200, __value)
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
        val __v = opName
        if (!__v.isEmpty) {
          _output__.writeString(1, __v)
        }
      };
      edgeType.foreach { __v =>
        val __m = __v
        _output__.writeTag(2, 2)
        _output__.writeUInt32NoTag(__m.serializedSize)
        __m.writeTo(_output__)
      };
      inputOpNames.foreach { __v =>
        val __m = __v
        _output__.writeString(3, __m)
      };
      samplingMethod.randomUniform.foreach { __v =>
        val __m = __v
        _output__.writeTag(100, 2)
        _output__.writeUInt32NoTag(__m.serializedSize)
        __m.writeTo(_output__)
      };
      samplingMethod.randomWeighted.foreach { __v =>
        val __m = __v
        _output__.writeTag(101, 2)
        _output__.writeUInt32NoTag(__m.serializedSize)
        __m.writeTo(_output__)
      };
      samplingMethod.topK.foreach { __v =>
        val __m = __v
        _output__.writeTag(103, 2)
        _output__.writeUInt32NoTag(__m.serializedSize)
        __m.writeTo(_output__)
      };
      samplingMethod.userDefined.foreach { __v =>
        val __m = __v
        _output__.writeTag(104, 2)
        _output__.writeUInt32NoTag(__m.serializedSize)
        __m.writeTo(_output__)
      };
      {
        val __v = samplingDirection.value
        if (__v != 0) {
          _output__.writeEnum(200, __v)
        }
      };
      unknownFields.writeTo(_output__)
    }
    def withOpName(__v: _root_.scala.Predef.String): SamplingOp = copy(opName = __v)
    def getEdgeType: snapchat.research.gbml.graph_schema.EdgeType = edgeType.getOrElse(snapchat.research.gbml.graph_schema.EdgeType.defaultInstance)
    def clearEdgeType: SamplingOp = copy(edgeType = _root_.scala.None)
    def withEdgeType(__v: snapchat.research.gbml.graph_schema.EdgeType): SamplingOp = copy(edgeType = Option(__v))
    def clearInputOpNames = copy(inputOpNames = _root_.scala.Seq.empty)
    def addInputOpNames(__vs: _root_.scala.Predef.String *): SamplingOp = addAllInputOpNames(__vs)
    def addAllInputOpNames(__vs: Iterable[_root_.scala.Predef.String]): SamplingOp = copy(inputOpNames = inputOpNames ++ __vs)
    def withInputOpNames(__v: _root_.scala.Seq[_root_.scala.Predef.String]): SamplingOp = copy(inputOpNames = __v)
    def getRandomUniform: snapchat.research.gbml.subgraph_sampling_strategy.RandomUniform = samplingMethod.randomUniform.getOrElse(snapchat.research.gbml.subgraph_sampling_strategy.RandomUniform.defaultInstance)
    def withRandomUniform(__v: snapchat.research.gbml.subgraph_sampling_strategy.RandomUniform): SamplingOp = copy(samplingMethod = snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod.RandomUniform(__v))
    def getRandomWeighted: snapchat.research.gbml.subgraph_sampling_strategy.RandomWeighted = samplingMethod.randomWeighted.getOrElse(snapchat.research.gbml.subgraph_sampling_strategy.RandomWeighted.defaultInstance)
    def withRandomWeighted(__v: snapchat.research.gbml.subgraph_sampling_strategy.RandomWeighted): SamplingOp = copy(samplingMethod = snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod.RandomWeighted(__v))
    def getTopK: snapchat.research.gbml.subgraph_sampling_strategy.TopK = samplingMethod.topK.getOrElse(snapchat.research.gbml.subgraph_sampling_strategy.TopK.defaultInstance)
    def withTopK(__v: snapchat.research.gbml.subgraph_sampling_strategy.TopK): SamplingOp = copy(samplingMethod = snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod.TopK(__v))
    def getUserDefined: snapchat.research.gbml.subgraph_sampling_strategy.UserDefined = samplingMethod.userDefined.getOrElse(snapchat.research.gbml.subgraph_sampling_strategy.UserDefined.defaultInstance)
    def withUserDefined(__v: snapchat.research.gbml.subgraph_sampling_strategy.UserDefined): SamplingOp = copy(samplingMethod = snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod.UserDefined(__v))
    def withSamplingDirection(__v: snapchat.research.gbml.subgraph_sampling_strategy.SamplingDirection): SamplingOp = copy(samplingDirection = __v)
    def clearSamplingMethod: SamplingOp = copy(samplingMethod = snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod.Empty)
    def withSamplingMethod(__v: snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod): SamplingOp = copy(samplingMethod = __v)
    def withUnknownFields(__v: _root_.scalapb.UnknownFieldSet) = copy(unknownFields = __v)
    def discardUnknownFields = copy(unknownFields = _root_.scalapb.UnknownFieldSet.empty)
    def getFieldByNumber(__fieldNumber: _root_.scala.Int): _root_.scala.Any = {
      (__fieldNumber: @_root_.scala.unchecked) match {
        case 1 => {
          val __t = opName
          if (__t != "") __t else null
        }
        case 2 => edgeType.orNull
        case 3 => inputOpNames
        case 100 => samplingMethod.randomUniform.orNull
        case 101 => samplingMethod.randomWeighted.orNull
        case 103 => samplingMethod.topK.orNull
        case 104 => samplingMethod.userDefined.orNull
        case 200 => {
          val __t = samplingDirection.javaValueDescriptor
          if (__t.getNumber() != 0) __t else null
        }
      }
    }
    def getField(__field: _root_.scalapb.descriptors.FieldDescriptor): _root_.scalapb.descriptors.PValue = {
      _root_.scala.Predef.require(__field.containingMessage eq companion.scalaDescriptor)
      (__field.number: @_root_.scala.unchecked) match {
        case 1 => _root_.scalapb.descriptors.PString(opName)
        case 2 => edgeType.map(_.toPMessage).getOrElse(_root_.scalapb.descriptors.PEmpty)
        case 3 => _root_.scalapb.descriptors.PRepeated(inputOpNames.iterator.map(_root_.scalapb.descriptors.PString(_)).toVector)
        case 100 => samplingMethod.randomUniform.map(_.toPMessage).getOrElse(_root_.scalapb.descriptors.PEmpty)
        case 101 => samplingMethod.randomWeighted.map(_.toPMessage).getOrElse(_root_.scalapb.descriptors.PEmpty)
        case 103 => samplingMethod.topK.map(_.toPMessage).getOrElse(_root_.scalapb.descriptors.PEmpty)
        case 104 => samplingMethod.userDefined.map(_.toPMessage).getOrElse(_root_.scalapb.descriptors.PEmpty)
        case 200 => _root_.scalapb.descriptors.PEnum(samplingDirection.scalaValueDescriptor)
      }
    }
    def toProtoString: _root_.scala.Predef.String = _root_.scalapb.TextFormat.printToUnicodeString(this)
    def companion: snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.type = snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp
    // @@protoc_insertion_point(GeneratedMessage[snapchat.research.gbml.SamplingOp])
}

object SamplingOp extends scalapb.GeneratedMessageCompanion[snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp] {
  implicit def messageCompanion: scalapb.GeneratedMessageCompanion[snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp] = this
  def parseFrom(`_input__`: _root_.com.google.protobuf.CodedInputStream): snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp = {
    var __opName: _root_.scala.Predef.String = ""
    var __edgeType: _root_.scala.Option[snapchat.research.gbml.graph_schema.EdgeType] = _root_.scala.None
    val __inputOpNames: _root_.scala.collection.immutable.VectorBuilder[_root_.scala.Predef.String] = new _root_.scala.collection.immutable.VectorBuilder[_root_.scala.Predef.String]
    var __samplingDirection: snapchat.research.gbml.subgraph_sampling_strategy.SamplingDirection = snapchat.research.gbml.subgraph_sampling_strategy.SamplingDirection.INCOMING
    var __samplingMethod: snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod = snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod.Empty
    var `_unknownFields__`: _root_.scalapb.UnknownFieldSet.Builder = null
    var _done__ = false
    while (!_done__) {
      val _tag__ = _input__.readTag()
      _tag__ match {
        case 0 => _done__ = true
        case 10 =>
          __opName = _input__.readStringRequireUtf8()
        case 18 =>
          __edgeType = Option(__edgeType.fold(_root_.scalapb.LiteParser.readMessage[snapchat.research.gbml.graph_schema.EdgeType](_input__))(_root_.scalapb.LiteParser.readMessage(_input__, _)))
        case 26 =>
          __inputOpNames += _input__.readStringRequireUtf8()
        case 802 =>
          __samplingMethod = snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod.RandomUniform(__samplingMethod.randomUniform.fold(_root_.scalapb.LiteParser.readMessage[snapchat.research.gbml.subgraph_sampling_strategy.RandomUniform](_input__))(_root_.scalapb.LiteParser.readMessage(_input__, _)))
        case 810 =>
          __samplingMethod = snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod.RandomWeighted(__samplingMethod.randomWeighted.fold(_root_.scalapb.LiteParser.readMessage[snapchat.research.gbml.subgraph_sampling_strategy.RandomWeighted](_input__))(_root_.scalapb.LiteParser.readMessage(_input__, _)))
        case 826 =>
          __samplingMethod = snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod.TopK(__samplingMethod.topK.fold(_root_.scalapb.LiteParser.readMessage[snapchat.research.gbml.subgraph_sampling_strategy.TopK](_input__))(_root_.scalapb.LiteParser.readMessage(_input__, _)))
        case 834 =>
          __samplingMethod = snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod.UserDefined(__samplingMethod.userDefined.fold(_root_.scalapb.LiteParser.readMessage[snapchat.research.gbml.subgraph_sampling_strategy.UserDefined](_input__))(_root_.scalapb.LiteParser.readMessage(_input__, _)))
        case 1600 =>
          __samplingDirection = snapchat.research.gbml.subgraph_sampling_strategy.SamplingDirection.fromValue(_input__.readEnum())
        case tag =>
          if (_unknownFields__ == null) {
            _unknownFields__ = new _root_.scalapb.UnknownFieldSet.Builder()
          }
          _unknownFields__.parseField(tag, _input__)
      }
    }
    snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp(
        opName = __opName,
        edgeType = __edgeType,
        inputOpNames = __inputOpNames.result(),
        samplingDirection = __samplingDirection,
        samplingMethod = __samplingMethod,
        unknownFields = if (_unknownFields__ == null) _root_.scalapb.UnknownFieldSet.empty else _unknownFields__.result()
    )
  }
  implicit def messageReads: _root_.scalapb.descriptors.Reads[snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp] = _root_.scalapb.descriptors.Reads{
    case _root_.scalapb.descriptors.PMessage(__fieldsMap) =>
      _root_.scala.Predef.require(__fieldsMap.keys.forall(_.containingMessage eq scalaDescriptor), "FieldDescriptor does not match message type.")
      snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp(
        opName = __fieldsMap.get(scalaDescriptor.findFieldByNumber(1).get).map(_.as[_root_.scala.Predef.String]).getOrElse(""),
        edgeType = __fieldsMap.get(scalaDescriptor.findFieldByNumber(2).get).flatMap(_.as[_root_.scala.Option[snapchat.research.gbml.graph_schema.EdgeType]]),
        inputOpNames = __fieldsMap.get(scalaDescriptor.findFieldByNumber(3).get).map(_.as[_root_.scala.Seq[_root_.scala.Predef.String]]).getOrElse(_root_.scala.Seq.empty),
        samplingDirection = snapchat.research.gbml.subgraph_sampling_strategy.SamplingDirection.fromValue(__fieldsMap.get(scalaDescriptor.findFieldByNumber(200).get).map(_.as[_root_.scalapb.descriptors.EnumValueDescriptor]).getOrElse(snapchat.research.gbml.subgraph_sampling_strategy.SamplingDirection.INCOMING.scalaValueDescriptor).number),
        samplingMethod = __fieldsMap.get(scalaDescriptor.findFieldByNumber(100).get).flatMap(_.as[_root_.scala.Option[snapchat.research.gbml.subgraph_sampling_strategy.RandomUniform]]).map(snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod.RandomUniform(_))
            .orElse[snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod](__fieldsMap.get(scalaDescriptor.findFieldByNumber(101).get).flatMap(_.as[_root_.scala.Option[snapchat.research.gbml.subgraph_sampling_strategy.RandomWeighted]]).map(snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod.RandomWeighted(_)))
            .orElse[snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod](__fieldsMap.get(scalaDescriptor.findFieldByNumber(103).get).flatMap(_.as[_root_.scala.Option[snapchat.research.gbml.subgraph_sampling_strategy.TopK]]).map(snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod.TopK(_)))
            .orElse[snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod](__fieldsMap.get(scalaDescriptor.findFieldByNumber(104).get).flatMap(_.as[_root_.scala.Option[snapchat.research.gbml.subgraph_sampling_strategy.UserDefined]]).map(snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod.UserDefined(_)))
            .getOrElse(snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod.Empty)
      )
    case _ => throw new RuntimeException("Expected PMessage")
  }
  def javaDescriptor: _root_.com.google.protobuf.Descriptors.Descriptor = SubgraphSamplingStrategyProto.javaDescriptor.getMessageTypes().get(4)
  def scalaDescriptor: _root_.scalapb.descriptors.Descriptor = SubgraphSamplingStrategyProto.scalaDescriptor.messages(4)
  def messageCompanionForFieldNumber(__number: _root_.scala.Int): _root_.scalapb.GeneratedMessageCompanion[_] = {
    var __out: _root_.scalapb.GeneratedMessageCompanion[_] = null
    (__number: @_root_.scala.unchecked) match {
      case 2 => __out = snapchat.research.gbml.graph_schema.EdgeType
      case 100 => __out = snapchat.research.gbml.subgraph_sampling_strategy.RandomUniform
      case 101 => __out = snapchat.research.gbml.subgraph_sampling_strategy.RandomWeighted
      case 103 => __out = snapchat.research.gbml.subgraph_sampling_strategy.TopK
      case 104 => __out = snapchat.research.gbml.subgraph_sampling_strategy.UserDefined
    }
    __out
  }
  lazy val nestedMessagesCompanions: Seq[_root_.scalapb.GeneratedMessageCompanion[_ <: _root_.scalapb.GeneratedMessage]] = Seq.empty
  def enumCompanionForFieldNumber(__fieldNumber: _root_.scala.Int): _root_.scalapb.GeneratedEnumCompanion[_] = {
    (__fieldNumber: @_root_.scala.unchecked) match {
      case 200 => snapchat.research.gbml.subgraph_sampling_strategy.SamplingDirection
    }
  }
  lazy val defaultInstance = snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp(
    opName = "",
    edgeType = _root_.scala.None,
    inputOpNames = _root_.scala.Seq.empty,
    samplingDirection = snapchat.research.gbml.subgraph_sampling_strategy.SamplingDirection.INCOMING,
    samplingMethod = snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod.Empty
  )
  sealed trait SamplingMethod extends _root_.scalapb.GeneratedOneof {
    def isEmpty: _root_.scala.Boolean = false
    def isDefined: _root_.scala.Boolean = true
    def isRandomUniform: _root_.scala.Boolean = false
    def isRandomWeighted: _root_.scala.Boolean = false
    def isTopK: _root_.scala.Boolean = false
    def isUserDefined: _root_.scala.Boolean = false
    def randomUniform: _root_.scala.Option[snapchat.research.gbml.subgraph_sampling_strategy.RandomUniform] = _root_.scala.None
    def randomWeighted: _root_.scala.Option[snapchat.research.gbml.subgraph_sampling_strategy.RandomWeighted] = _root_.scala.None
    def topK: _root_.scala.Option[snapchat.research.gbml.subgraph_sampling_strategy.TopK] = _root_.scala.None
    def userDefined: _root_.scala.Option[snapchat.research.gbml.subgraph_sampling_strategy.UserDefined] = _root_.scala.None
  }
  object SamplingMethod {
    @SerialVersionUID(0L)
    case object Empty extends snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod {
      type ValueType = _root_.scala.Nothing
      override def isEmpty: _root_.scala.Boolean = true
      override def isDefined: _root_.scala.Boolean = false
      override def number: _root_.scala.Int = 0
      override def value: _root_.scala.Nothing = throw new java.util.NoSuchElementException("Empty.value")
    }
  
    @SerialVersionUID(0L)
    final case class RandomUniform(value: snapchat.research.gbml.subgraph_sampling_strategy.RandomUniform) extends snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod {
      type ValueType = snapchat.research.gbml.subgraph_sampling_strategy.RandomUniform
      override def isRandomUniform: _root_.scala.Boolean = true
      override def randomUniform: _root_.scala.Option[snapchat.research.gbml.subgraph_sampling_strategy.RandomUniform] = Some(value)
      override def number: _root_.scala.Int = 100
    }
    @SerialVersionUID(0L)
    final case class RandomWeighted(value: snapchat.research.gbml.subgraph_sampling_strategy.RandomWeighted) extends snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod {
      type ValueType = snapchat.research.gbml.subgraph_sampling_strategy.RandomWeighted
      override def isRandomWeighted: _root_.scala.Boolean = true
      override def randomWeighted: _root_.scala.Option[snapchat.research.gbml.subgraph_sampling_strategy.RandomWeighted] = Some(value)
      override def number: _root_.scala.Int = 101
    }
    @SerialVersionUID(0L)
    final case class TopK(value: snapchat.research.gbml.subgraph_sampling_strategy.TopK) extends snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod {
      type ValueType = snapchat.research.gbml.subgraph_sampling_strategy.TopK
      override def isTopK: _root_.scala.Boolean = true
      override def topK: _root_.scala.Option[snapchat.research.gbml.subgraph_sampling_strategy.TopK] = Some(value)
      override def number: _root_.scala.Int = 103
    }
    @SerialVersionUID(0L)
    final case class UserDefined(value: snapchat.research.gbml.subgraph_sampling_strategy.UserDefined) extends snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod {
      type ValueType = snapchat.research.gbml.subgraph_sampling_strategy.UserDefined
      override def isUserDefined: _root_.scala.Boolean = true
      override def userDefined: _root_.scala.Option[snapchat.research.gbml.subgraph_sampling_strategy.UserDefined] = Some(value)
      override def number: _root_.scala.Int = 104
    }
  }
  implicit class SamplingOpLens[UpperPB](_l: _root_.scalapb.lenses.Lens[UpperPB, snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp]) extends _root_.scalapb.lenses.ObjectLens[UpperPB, snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp](_l) {
    def opName: _root_.scalapb.lenses.Lens[UpperPB, _root_.scala.Predef.String] = field(_.opName)((c_, f_) => c_.copy(opName = f_))
    def edgeType: _root_.scalapb.lenses.Lens[UpperPB, snapchat.research.gbml.graph_schema.EdgeType] = field(_.getEdgeType)((c_, f_) => c_.copy(edgeType = Option(f_)))
    def optionalEdgeType: _root_.scalapb.lenses.Lens[UpperPB, _root_.scala.Option[snapchat.research.gbml.graph_schema.EdgeType]] = field(_.edgeType)((c_, f_) => c_.copy(edgeType = f_))
    def inputOpNames: _root_.scalapb.lenses.Lens[UpperPB, _root_.scala.Seq[_root_.scala.Predef.String]] = field(_.inputOpNames)((c_, f_) => c_.copy(inputOpNames = f_))
    def randomUniform: _root_.scalapb.lenses.Lens[UpperPB, snapchat.research.gbml.subgraph_sampling_strategy.RandomUniform] = field(_.getRandomUniform)((c_, f_) => c_.copy(samplingMethod = snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod.RandomUniform(f_)))
    def randomWeighted: _root_.scalapb.lenses.Lens[UpperPB, snapchat.research.gbml.subgraph_sampling_strategy.RandomWeighted] = field(_.getRandomWeighted)((c_, f_) => c_.copy(samplingMethod = snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod.RandomWeighted(f_)))
    def topK: _root_.scalapb.lenses.Lens[UpperPB, snapchat.research.gbml.subgraph_sampling_strategy.TopK] = field(_.getTopK)((c_, f_) => c_.copy(samplingMethod = snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod.TopK(f_)))
    def userDefined: _root_.scalapb.lenses.Lens[UpperPB, snapchat.research.gbml.subgraph_sampling_strategy.UserDefined] = field(_.getUserDefined)((c_, f_) => c_.copy(samplingMethod = snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod.UserDefined(f_)))
    def samplingDirection: _root_.scalapb.lenses.Lens[UpperPB, snapchat.research.gbml.subgraph_sampling_strategy.SamplingDirection] = field(_.samplingDirection)((c_, f_) => c_.copy(samplingDirection = f_))
    def samplingMethod: _root_.scalapb.lenses.Lens[UpperPB, snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod] = field(_.samplingMethod)((c_, f_) => c_.copy(samplingMethod = f_))
  }
  final val OP_NAME_FIELD_NUMBER = 1
  final val EDGE_TYPE_FIELD_NUMBER = 2
  final val INPUT_OP_NAMES_FIELD_NUMBER = 3
  final val RANDOM_UNIFORM_FIELD_NUMBER = 100
  final val RANDOM_WEIGHTED_FIELD_NUMBER = 101
  final val TOP_K_FIELD_NUMBER = 103
  final val USER_DEFINED_FIELD_NUMBER = 104
  final val SAMPLING_DIRECTION_FIELD_NUMBER = 200
  def of(
    opName: _root_.scala.Predef.String,
    edgeType: _root_.scala.Option[snapchat.research.gbml.graph_schema.EdgeType],
    inputOpNames: _root_.scala.Seq[_root_.scala.Predef.String],
    samplingMethod: snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp.SamplingMethod,
    samplingDirection: snapchat.research.gbml.subgraph_sampling_strategy.SamplingDirection
  ): _root_.snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp = _root_.snapchat.research.gbml.subgraph_sampling_strategy.SamplingOp(
    opName,
    edgeType,
    inputOpNames,
    samplingMethod,
    samplingDirection
  )
  // @@protoc_insertion_point(GeneratedMessageCompanion[snapchat.research.gbml.SamplingOp])
}
