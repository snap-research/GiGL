// Generated by the Scala Plugin for the Protocol Buffer Compiler.
// Do not edit!
//
// Protofile syntax: PROTO3

package snapchat.research.gbml.gigl_resource_config

/** Shared resources configuration
  */
@SerialVersionUID(0L)
final case class SharedResourceConfig(
    resourceLabels: _root_.scala.collection.immutable.Map[_root_.scala.Predef.String, _root_.scala.Predef.String] = _root_.scala.collection.immutable.Map.empty,
    commonComputeConfig: _root_.scala.Option[snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.CommonComputeConfig] = _root_.scala.None,
    unknownFields: _root_.scalapb.UnknownFieldSet = _root_.scalapb.UnknownFieldSet.empty
    ) extends scalapb.GeneratedMessage with scalapb.lenses.Updatable[SharedResourceConfig] {
    @transient
    private[this] var __serializedSizeMemoized: _root_.scala.Int = 0
    private[this] def __computeSerializedSize(): _root_.scala.Int = {
      var __size = 0
      resourceLabels.foreach { __item =>
        val __value = snapchat.research.gbml.gigl_resource_config.SharedResourceConfig._typemapper_resourceLabels.toBase(__item)
        __size += 1 + _root_.com.google.protobuf.CodedOutputStream.computeUInt32SizeNoTag(__value.serializedSize) + __value.serializedSize
      }
      if (commonComputeConfig.isDefined) {
        val __value = commonComputeConfig.get
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
      resourceLabels.foreach { __v =>
        val __m = snapchat.research.gbml.gigl_resource_config.SharedResourceConfig._typemapper_resourceLabels.toBase(__v)
        _output__.writeTag(1, 2)
        _output__.writeUInt32NoTag(__m.serializedSize)
        __m.writeTo(_output__)
      };
      commonComputeConfig.foreach { __v =>
        val __m = __v
        _output__.writeTag(2, 2)
        _output__.writeUInt32NoTag(__m.serializedSize)
        __m.writeTo(_output__)
      };
      unknownFields.writeTo(_output__)
    }
    def clearResourceLabels = copy(resourceLabels = _root_.scala.collection.immutable.Map.empty)
    def addResourceLabels(__vs: (_root_.scala.Predef.String, _root_.scala.Predef.String) *): SharedResourceConfig = addAllResourceLabels(__vs)
    def addAllResourceLabels(__vs: Iterable[(_root_.scala.Predef.String, _root_.scala.Predef.String)]): SharedResourceConfig = copy(resourceLabels = resourceLabels ++ __vs)
    def withResourceLabels(__v: _root_.scala.collection.immutable.Map[_root_.scala.Predef.String, _root_.scala.Predef.String]): SharedResourceConfig = copy(resourceLabels = __v)
    def getCommonComputeConfig: snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.CommonComputeConfig = commonComputeConfig.getOrElse(snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.CommonComputeConfig.defaultInstance)
    def clearCommonComputeConfig: SharedResourceConfig = copy(commonComputeConfig = _root_.scala.None)
    def withCommonComputeConfig(__v: snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.CommonComputeConfig): SharedResourceConfig = copy(commonComputeConfig = Option(__v))
    def withUnknownFields(__v: _root_.scalapb.UnknownFieldSet) = copy(unknownFields = __v)
    def discardUnknownFields = copy(unknownFields = _root_.scalapb.UnknownFieldSet.empty)
    def getFieldByNumber(__fieldNumber: _root_.scala.Int): _root_.scala.Any = {
      (__fieldNumber: @_root_.scala.unchecked) match {
        case 1 => resourceLabels.iterator.map(snapchat.research.gbml.gigl_resource_config.SharedResourceConfig._typemapper_resourceLabels.toBase(_)).toSeq
        case 2 => commonComputeConfig.orNull
      }
    }
    def getField(__field: _root_.scalapb.descriptors.FieldDescriptor): _root_.scalapb.descriptors.PValue = {
      _root_.scala.Predef.require(__field.containingMessage eq companion.scalaDescriptor)
      (__field.number: @_root_.scala.unchecked) match {
        case 1 => _root_.scalapb.descriptors.PRepeated(resourceLabels.iterator.map(snapchat.research.gbml.gigl_resource_config.SharedResourceConfig._typemapper_resourceLabels.toBase(_).toPMessage).toVector)
        case 2 => commonComputeConfig.map(_.toPMessage).getOrElse(_root_.scalapb.descriptors.PEmpty)
      }
    }
    def toProtoString: _root_.scala.Predef.String = _root_.scalapb.TextFormat.printToUnicodeString(this)
    def companion: snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.type = snapchat.research.gbml.gigl_resource_config.SharedResourceConfig
    // @@protoc_insertion_point(GeneratedMessage[snapchat.research.gbml.SharedResourceConfig])
}

object SharedResourceConfig extends scalapb.GeneratedMessageCompanion[snapchat.research.gbml.gigl_resource_config.SharedResourceConfig] {
  implicit def messageCompanion: scalapb.GeneratedMessageCompanion[snapchat.research.gbml.gigl_resource_config.SharedResourceConfig] = this
  def parseFrom(`_input__`: _root_.com.google.protobuf.CodedInputStream): snapchat.research.gbml.gigl_resource_config.SharedResourceConfig = {
    val __resourceLabels: _root_.scala.collection.mutable.Builder[(_root_.scala.Predef.String, _root_.scala.Predef.String), _root_.scala.collection.immutable.Map[_root_.scala.Predef.String, _root_.scala.Predef.String]] = _root_.scala.collection.immutable.Map.newBuilder[_root_.scala.Predef.String, _root_.scala.Predef.String]
    var __commonComputeConfig: _root_.scala.Option[snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.CommonComputeConfig] = _root_.scala.None
    var `_unknownFields__`: _root_.scalapb.UnknownFieldSet.Builder = null
    var _done__ = false
    while (!_done__) {
      val _tag__ = _input__.readTag()
      _tag__ match {
        case 0 => _done__ = true
        case 10 =>
          __resourceLabels += snapchat.research.gbml.gigl_resource_config.SharedResourceConfig._typemapper_resourceLabels.toCustom(_root_.scalapb.LiteParser.readMessage[snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.ResourceLabelsEntry](_input__))
        case 18 =>
          __commonComputeConfig = Option(__commonComputeConfig.fold(_root_.scalapb.LiteParser.readMessage[snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.CommonComputeConfig](_input__))(_root_.scalapb.LiteParser.readMessage(_input__, _)))
        case tag =>
          if (_unknownFields__ == null) {
            _unknownFields__ = new _root_.scalapb.UnknownFieldSet.Builder()
          }
          _unknownFields__.parseField(tag, _input__)
      }
    }
    snapchat.research.gbml.gigl_resource_config.SharedResourceConfig(
        resourceLabels = __resourceLabels.result(),
        commonComputeConfig = __commonComputeConfig,
        unknownFields = if (_unknownFields__ == null) _root_.scalapb.UnknownFieldSet.empty else _unknownFields__.result()
    )
  }
  implicit def messageReads: _root_.scalapb.descriptors.Reads[snapchat.research.gbml.gigl_resource_config.SharedResourceConfig] = _root_.scalapb.descriptors.Reads{
    case _root_.scalapb.descriptors.PMessage(__fieldsMap) =>
      _root_.scala.Predef.require(__fieldsMap.keys.forall(_.containingMessage eq scalaDescriptor), "FieldDescriptor does not match message type.")
      snapchat.research.gbml.gigl_resource_config.SharedResourceConfig(
        resourceLabels = __fieldsMap.get(scalaDescriptor.findFieldByNumber(1).get).map(_.as[_root_.scala.Seq[snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.ResourceLabelsEntry]]).getOrElse(_root_.scala.Seq.empty).iterator.map(snapchat.research.gbml.gigl_resource_config.SharedResourceConfig._typemapper_resourceLabels.toCustom(_)).toMap,
        commonComputeConfig = __fieldsMap.get(scalaDescriptor.findFieldByNumber(2).get).flatMap(_.as[_root_.scala.Option[snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.CommonComputeConfig]])
      )
    case _ => throw new RuntimeException("Expected PMessage")
  }
  def javaDescriptor: _root_.com.google.protobuf.Descriptors.Descriptor = GiglResourceConfigProto.javaDescriptor.getMessageTypes().get(12)
  def scalaDescriptor: _root_.scalapb.descriptors.Descriptor = GiglResourceConfigProto.scalaDescriptor.messages(12)
  def messageCompanionForFieldNumber(__number: _root_.scala.Int): _root_.scalapb.GeneratedMessageCompanion[_] = {
    var __out: _root_.scalapb.GeneratedMessageCompanion[_] = null
    (__number: @_root_.scala.unchecked) match {
      case 1 => __out = snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.ResourceLabelsEntry
      case 2 => __out = snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.CommonComputeConfig
    }
    __out
  }
  lazy val nestedMessagesCompanions: Seq[_root_.scalapb.GeneratedMessageCompanion[_ <: _root_.scalapb.GeneratedMessage]] =
    Seq[_root_.scalapb.GeneratedMessageCompanion[_ <: _root_.scalapb.GeneratedMessage]](
      _root_.snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.CommonComputeConfig,
      _root_.snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.ResourceLabelsEntry
    )
  def enumCompanionForFieldNumber(__fieldNumber: _root_.scala.Int): _root_.scalapb.GeneratedEnumCompanion[_] = throw new MatchError(__fieldNumber)
  lazy val defaultInstance = snapchat.research.gbml.gigl_resource_config.SharedResourceConfig(
    resourceLabels = _root_.scala.collection.immutable.Map.empty,
    commonComputeConfig = _root_.scala.None
  )
  /** @param project
    *   GCP Project
    * @param region
    *   GCP Region where compute is to be scheduled
    * @param tempAssetsBucket
    *   GCS Bucket for where temporary assets are to be stored
    * @param tempRegionalAssetsBucket
    *   Regional GCS Bucket used to store temporary assets
    * @param permAssetsBucket
    *   Regional GCS Bucket that will store permanent assets like Trained Model
    * @param tempAssetsBqDatasetName
    *   Path to BQ dataset used to store temporary assets
    * @param embeddingBqDatasetName
    *   Path to BQ Dataset used to persist generated embeddings and predictions
    * @param gcpServiceAccountEmail
    *   The GCP service account email being used to schedule compute on GCP
    * @param dataflowRunner
    *   The runner to use for Dataflow i.e DirectRunner or DataflowRunner
    */
  @SerialVersionUID(0L)
  final case class CommonComputeConfig(
      project: _root_.scala.Predef.String = "",
      region: _root_.scala.Predef.String = "",
      tempAssetsBucket: _root_.scala.Predef.String = "",
      tempRegionalAssetsBucket: _root_.scala.Predef.String = "",
      permAssetsBucket: _root_.scala.Predef.String = "",
      tempAssetsBqDatasetName: _root_.scala.Predef.String = "",
      embeddingBqDatasetName: _root_.scala.Predef.String = "",
      gcpServiceAccountEmail: _root_.scala.Predef.String = "",
      dataflowRunner: _root_.scala.Predef.String = "",
      unknownFields: _root_.scalapb.UnknownFieldSet = _root_.scalapb.UnknownFieldSet.empty
      ) extends scalapb.GeneratedMessage with scalapb.lenses.Updatable[CommonComputeConfig] {
      @transient
      private[this] var __serializedSizeMemoized: _root_.scala.Int = 0
      private[this] def __computeSerializedSize(): _root_.scala.Int = {
        var __size = 0
        
        {
          val __value = project
          if (!__value.isEmpty) {
            __size += _root_.com.google.protobuf.CodedOutputStream.computeStringSize(1, __value)
          }
        };
        
        {
          val __value = region
          if (!__value.isEmpty) {
            __size += _root_.com.google.protobuf.CodedOutputStream.computeStringSize(2, __value)
          }
        };
        
        {
          val __value = tempAssetsBucket
          if (!__value.isEmpty) {
            __size += _root_.com.google.protobuf.CodedOutputStream.computeStringSize(3, __value)
          }
        };
        
        {
          val __value = tempRegionalAssetsBucket
          if (!__value.isEmpty) {
            __size += _root_.com.google.protobuf.CodedOutputStream.computeStringSize(4, __value)
          }
        };
        
        {
          val __value = permAssetsBucket
          if (!__value.isEmpty) {
            __size += _root_.com.google.protobuf.CodedOutputStream.computeStringSize(5, __value)
          }
        };
        
        {
          val __value = tempAssetsBqDatasetName
          if (!__value.isEmpty) {
            __size += _root_.com.google.protobuf.CodedOutputStream.computeStringSize(6, __value)
          }
        };
        
        {
          val __value = embeddingBqDatasetName
          if (!__value.isEmpty) {
            __size += _root_.com.google.protobuf.CodedOutputStream.computeStringSize(7, __value)
          }
        };
        
        {
          val __value = gcpServiceAccountEmail
          if (!__value.isEmpty) {
            __size += _root_.com.google.protobuf.CodedOutputStream.computeStringSize(8, __value)
          }
        };
        
        {
          val __value = dataflowRunner
          if (!__value.isEmpty) {
            __size += _root_.com.google.protobuf.CodedOutputStream.computeStringSize(11, __value)
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
          val __v = project
          if (!__v.isEmpty) {
            _output__.writeString(1, __v)
          }
        };
        {
          val __v = region
          if (!__v.isEmpty) {
            _output__.writeString(2, __v)
          }
        };
        {
          val __v = tempAssetsBucket
          if (!__v.isEmpty) {
            _output__.writeString(3, __v)
          }
        };
        {
          val __v = tempRegionalAssetsBucket
          if (!__v.isEmpty) {
            _output__.writeString(4, __v)
          }
        };
        {
          val __v = permAssetsBucket
          if (!__v.isEmpty) {
            _output__.writeString(5, __v)
          }
        };
        {
          val __v = tempAssetsBqDatasetName
          if (!__v.isEmpty) {
            _output__.writeString(6, __v)
          }
        };
        {
          val __v = embeddingBqDatasetName
          if (!__v.isEmpty) {
            _output__.writeString(7, __v)
          }
        };
        {
          val __v = gcpServiceAccountEmail
          if (!__v.isEmpty) {
            _output__.writeString(8, __v)
          }
        };
        {
          val __v = dataflowRunner
          if (!__v.isEmpty) {
            _output__.writeString(11, __v)
          }
        };
        unknownFields.writeTo(_output__)
      }
      def withProject(__v: _root_.scala.Predef.String): CommonComputeConfig = copy(project = __v)
      def withRegion(__v: _root_.scala.Predef.String): CommonComputeConfig = copy(region = __v)
      def withTempAssetsBucket(__v: _root_.scala.Predef.String): CommonComputeConfig = copy(tempAssetsBucket = __v)
      def withTempRegionalAssetsBucket(__v: _root_.scala.Predef.String): CommonComputeConfig = copy(tempRegionalAssetsBucket = __v)
      def withPermAssetsBucket(__v: _root_.scala.Predef.String): CommonComputeConfig = copy(permAssetsBucket = __v)
      def withTempAssetsBqDatasetName(__v: _root_.scala.Predef.String): CommonComputeConfig = copy(tempAssetsBqDatasetName = __v)
      def withEmbeddingBqDatasetName(__v: _root_.scala.Predef.String): CommonComputeConfig = copy(embeddingBqDatasetName = __v)
      def withGcpServiceAccountEmail(__v: _root_.scala.Predef.String): CommonComputeConfig = copy(gcpServiceAccountEmail = __v)
      def withDataflowRunner(__v: _root_.scala.Predef.String): CommonComputeConfig = copy(dataflowRunner = __v)
      def withUnknownFields(__v: _root_.scalapb.UnknownFieldSet) = copy(unknownFields = __v)
      def discardUnknownFields = copy(unknownFields = _root_.scalapb.UnknownFieldSet.empty)
      def getFieldByNumber(__fieldNumber: _root_.scala.Int): _root_.scala.Any = {
        (__fieldNumber: @_root_.scala.unchecked) match {
          case 1 => {
            val __t = project
            if (__t != "") __t else null
          }
          case 2 => {
            val __t = region
            if (__t != "") __t else null
          }
          case 3 => {
            val __t = tempAssetsBucket
            if (__t != "") __t else null
          }
          case 4 => {
            val __t = tempRegionalAssetsBucket
            if (__t != "") __t else null
          }
          case 5 => {
            val __t = permAssetsBucket
            if (__t != "") __t else null
          }
          case 6 => {
            val __t = tempAssetsBqDatasetName
            if (__t != "") __t else null
          }
          case 7 => {
            val __t = embeddingBqDatasetName
            if (__t != "") __t else null
          }
          case 8 => {
            val __t = gcpServiceAccountEmail
            if (__t != "") __t else null
          }
          case 11 => {
            val __t = dataflowRunner
            if (__t != "") __t else null
          }
        }
      }
      def getField(__field: _root_.scalapb.descriptors.FieldDescriptor): _root_.scalapb.descriptors.PValue = {
        _root_.scala.Predef.require(__field.containingMessage eq companion.scalaDescriptor)
        (__field.number: @_root_.scala.unchecked) match {
          case 1 => _root_.scalapb.descriptors.PString(project)
          case 2 => _root_.scalapb.descriptors.PString(region)
          case 3 => _root_.scalapb.descriptors.PString(tempAssetsBucket)
          case 4 => _root_.scalapb.descriptors.PString(tempRegionalAssetsBucket)
          case 5 => _root_.scalapb.descriptors.PString(permAssetsBucket)
          case 6 => _root_.scalapb.descriptors.PString(tempAssetsBqDatasetName)
          case 7 => _root_.scalapb.descriptors.PString(embeddingBqDatasetName)
          case 8 => _root_.scalapb.descriptors.PString(gcpServiceAccountEmail)
          case 11 => _root_.scalapb.descriptors.PString(dataflowRunner)
        }
      }
      def toProtoString: _root_.scala.Predef.String = _root_.scalapb.TextFormat.printToUnicodeString(this)
      def companion: snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.CommonComputeConfig.type = snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.CommonComputeConfig
      // @@protoc_insertion_point(GeneratedMessage[snapchat.research.gbml.SharedResourceConfig.CommonComputeConfig])
  }
  
  object CommonComputeConfig extends scalapb.GeneratedMessageCompanion[snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.CommonComputeConfig] {
    implicit def messageCompanion: scalapb.GeneratedMessageCompanion[snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.CommonComputeConfig] = this
    def parseFrom(`_input__`: _root_.com.google.protobuf.CodedInputStream): snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.CommonComputeConfig = {
      var __project: _root_.scala.Predef.String = ""
      var __region: _root_.scala.Predef.String = ""
      var __tempAssetsBucket: _root_.scala.Predef.String = ""
      var __tempRegionalAssetsBucket: _root_.scala.Predef.String = ""
      var __permAssetsBucket: _root_.scala.Predef.String = ""
      var __tempAssetsBqDatasetName: _root_.scala.Predef.String = ""
      var __embeddingBqDatasetName: _root_.scala.Predef.String = ""
      var __gcpServiceAccountEmail: _root_.scala.Predef.String = ""
      var __dataflowRunner: _root_.scala.Predef.String = ""
      var `_unknownFields__`: _root_.scalapb.UnknownFieldSet.Builder = null
      var _done__ = false
      while (!_done__) {
        val _tag__ = _input__.readTag()
        _tag__ match {
          case 0 => _done__ = true
          case 10 =>
            __project = _input__.readStringRequireUtf8()
          case 18 =>
            __region = _input__.readStringRequireUtf8()
          case 26 =>
            __tempAssetsBucket = _input__.readStringRequireUtf8()
          case 34 =>
            __tempRegionalAssetsBucket = _input__.readStringRequireUtf8()
          case 42 =>
            __permAssetsBucket = _input__.readStringRequireUtf8()
          case 50 =>
            __tempAssetsBqDatasetName = _input__.readStringRequireUtf8()
          case 58 =>
            __embeddingBqDatasetName = _input__.readStringRequireUtf8()
          case 66 =>
            __gcpServiceAccountEmail = _input__.readStringRequireUtf8()
          case 90 =>
            __dataflowRunner = _input__.readStringRequireUtf8()
          case tag =>
            if (_unknownFields__ == null) {
              _unknownFields__ = new _root_.scalapb.UnknownFieldSet.Builder()
            }
            _unknownFields__.parseField(tag, _input__)
        }
      }
      snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.CommonComputeConfig(
          project = __project,
          region = __region,
          tempAssetsBucket = __tempAssetsBucket,
          tempRegionalAssetsBucket = __tempRegionalAssetsBucket,
          permAssetsBucket = __permAssetsBucket,
          tempAssetsBqDatasetName = __tempAssetsBqDatasetName,
          embeddingBqDatasetName = __embeddingBqDatasetName,
          gcpServiceAccountEmail = __gcpServiceAccountEmail,
          dataflowRunner = __dataflowRunner,
          unknownFields = if (_unknownFields__ == null) _root_.scalapb.UnknownFieldSet.empty else _unknownFields__.result()
      )
    }
    implicit def messageReads: _root_.scalapb.descriptors.Reads[snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.CommonComputeConfig] = _root_.scalapb.descriptors.Reads{
      case _root_.scalapb.descriptors.PMessage(__fieldsMap) =>
        _root_.scala.Predef.require(__fieldsMap.keys.forall(_.containingMessage eq scalaDescriptor), "FieldDescriptor does not match message type.")
        snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.CommonComputeConfig(
          project = __fieldsMap.get(scalaDescriptor.findFieldByNumber(1).get).map(_.as[_root_.scala.Predef.String]).getOrElse(""),
          region = __fieldsMap.get(scalaDescriptor.findFieldByNumber(2).get).map(_.as[_root_.scala.Predef.String]).getOrElse(""),
          tempAssetsBucket = __fieldsMap.get(scalaDescriptor.findFieldByNumber(3).get).map(_.as[_root_.scala.Predef.String]).getOrElse(""),
          tempRegionalAssetsBucket = __fieldsMap.get(scalaDescriptor.findFieldByNumber(4).get).map(_.as[_root_.scala.Predef.String]).getOrElse(""),
          permAssetsBucket = __fieldsMap.get(scalaDescriptor.findFieldByNumber(5).get).map(_.as[_root_.scala.Predef.String]).getOrElse(""),
          tempAssetsBqDatasetName = __fieldsMap.get(scalaDescriptor.findFieldByNumber(6).get).map(_.as[_root_.scala.Predef.String]).getOrElse(""),
          embeddingBqDatasetName = __fieldsMap.get(scalaDescriptor.findFieldByNumber(7).get).map(_.as[_root_.scala.Predef.String]).getOrElse(""),
          gcpServiceAccountEmail = __fieldsMap.get(scalaDescriptor.findFieldByNumber(8).get).map(_.as[_root_.scala.Predef.String]).getOrElse(""),
          dataflowRunner = __fieldsMap.get(scalaDescriptor.findFieldByNumber(11).get).map(_.as[_root_.scala.Predef.String]).getOrElse("")
        )
      case _ => throw new RuntimeException("Expected PMessage")
    }
    def javaDescriptor: _root_.com.google.protobuf.Descriptors.Descriptor = snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.javaDescriptor.getNestedTypes().get(0)
    def scalaDescriptor: _root_.scalapb.descriptors.Descriptor = snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.scalaDescriptor.nestedMessages(0)
    def messageCompanionForFieldNumber(__number: _root_.scala.Int): _root_.scalapb.GeneratedMessageCompanion[_] = throw new MatchError(__number)
    lazy val nestedMessagesCompanions: Seq[_root_.scalapb.GeneratedMessageCompanion[_ <: _root_.scalapb.GeneratedMessage]] = Seq.empty
    def enumCompanionForFieldNumber(__fieldNumber: _root_.scala.Int): _root_.scalapb.GeneratedEnumCompanion[_] = throw new MatchError(__fieldNumber)
    lazy val defaultInstance = snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.CommonComputeConfig(
      project = "",
      region = "",
      tempAssetsBucket = "",
      tempRegionalAssetsBucket = "",
      permAssetsBucket = "",
      tempAssetsBqDatasetName = "",
      embeddingBqDatasetName = "",
      gcpServiceAccountEmail = "",
      dataflowRunner = ""
    )
    implicit class CommonComputeConfigLens[UpperPB](_l: _root_.scalapb.lenses.Lens[UpperPB, snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.CommonComputeConfig]) extends _root_.scalapb.lenses.ObjectLens[UpperPB, snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.CommonComputeConfig](_l) {
      def project: _root_.scalapb.lenses.Lens[UpperPB, _root_.scala.Predef.String] = field(_.project)((c_, f_) => c_.copy(project = f_))
      def region: _root_.scalapb.lenses.Lens[UpperPB, _root_.scala.Predef.String] = field(_.region)((c_, f_) => c_.copy(region = f_))
      def tempAssetsBucket: _root_.scalapb.lenses.Lens[UpperPB, _root_.scala.Predef.String] = field(_.tempAssetsBucket)((c_, f_) => c_.copy(tempAssetsBucket = f_))
      def tempRegionalAssetsBucket: _root_.scalapb.lenses.Lens[UpperPB, _root_.scala.Predef.String] = field(_.tempRegionalAssetsBucket)((c_, f_) => c_.copy(tempRegionalAssetsBucket = f_))
      def permAssetsBucket: _root_.scalapb.lenses.Lens[UpperPB, _root_.scala.Predef.String] = field(_.permAssetsBucket)((c_, f_) => c_.copy(permAssetsBucket = f_))
      def tempAssetsBqDatasetName: _root_.scalapb.lenses.Lens[UpperPB, _root_.scala.Predef.String] = field(_.tempAssetsBqDatasetName)((c_, f_) => c_.copy(tempAssetsBqDatasetName = f_))
      def embeddingBqDatasetName: _root_.scalapb.lenses.Lens[UpperPB, _root_.scala.Predef.String] = field(_.embeddingBqDatasetName)((c_, f_) => c_.copy(embeddingBqDatasetName = f_))
      def gcpServiceAccountEmail: _root_.scalapb.lenses.Lens[UpperPB, _root_.scala.Predef.String] = field(_.gcpServiceAccountEmail)((c_, f_) => c_.copy(gcpServiceAccountEmail = f_))
      def dataflowRunner: _root_.scalapb.lenses.Lens[UpperPB, _root_.scala.Predef.String] = field(_.dataflowRunner)((c_, f_) => c_.copy(dataflowRunner = f_))
    }
    final val PROJECT_FIELD_NUMBER = 1
    final val REGION_FIELD_NUMBER = 2
    final val TEMP_ASSETS_BUCKET_FIELD_NUMBER = 3
    final val TEMP_REGIONAL_ASSETS_BUCKET_FIELD_NUMBER = 4
    final val PERM_ASSETS_BUCKET_FIELD_NUMBER = 5
    final val TEMP_ASSETS_BQ_DATASET_NAME_FIELD_NUMBER = 6
    final val EMBEDDING_BQ_DATASET_NAME_FIELD_NUMBER = 7
    final val GCP_SERVICE_ACCOUNT_EMAIL_FIELD_NUMBER = 8
    final val DATAFLOW_RUNNER_FIELD_NUMBER = 11
    def of(
      project: _root_.scala.Predef.String,
      region: _root_.scala.Predef.String,
      tempAssetsBucket: _root_.scala.Predef.String,
      tempRegionalAssetsBucket: _root_.scala.Predef.String,
      permAssetsBucket: _root_.scala.Predef.String,
      tempAssetsBqDatasetName: _root_.scala.Predef.String,
      embeddingBqDatasetName: _root_.scala.Predef.String,
      gcpServiceAccountEmail: _root_.scala.Predef.String,
      dataflowRunner: _root_.scala.Predef.String
    ): _root_.snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.CommonComputeConfig = _root_.snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.CommonComputeConfig(
      project,
      region,
      tempAssetsBucket,
      tempRegionalAssetsBucket,
      permAssetsBucket,
      tempAssetsBqDatasetName,
      embeddingBqDatasetName,
      gcpServiceAccountEmail,
      dataflowRunner
    )
    // @@protoc_insertion_point(GeneratedMessageCompanion[snapchat.research.gbml.SharedResourceConfig.CommonComputeConfig])
  }
  
  @SerialVersionUID(0L)
  final case class ResourceLabelsEntry(
      key: _root_.scala.Predef.String = "",
      value: _root_.scala.Predef.String = "",
      unknownFields: _root_.scalapb.UnknownFieldSet = _root_.scalapb.UnknownFieldSet.empty
      ) extends scalapb.GeneratedMessage with scalapb.lenses.Updatable[ResourceLabelsEntry] {
      @transient
      private[this] var __serializedSizeMemoized: _root_.scala.Int = 0
      private[this] def __computeSerializedSize(): _root_.scala.Int = {
        var __size = 0
        
        {
          val __value = key
          if (!__value.isEmpty) {
            __size += _root_.com.google.protobuf.CodedOutputStream.computeStringSize(1, __value)
          }
        };
        
        {
          val __value = value
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
          val __v = key
          if (!__v.isEmpty) {
            _output__.writeString(1, __v)
          }
        };
        {
          val __v = value
          if (!__v.isEmpty) {
            _output__.writeString(2, __v)
          }
        };
        unknownFields.writeTo(_output__)
      }
      def withKey(__v: _root_.scala.Predef.String): ResourceLabelsEntry = copy(key = __v)
      def withValue(__v: _root_.scala.Predef.String): ResourceLabelsEntry = copy(value = __v)
      def withUnknownFields(__v: _root_.scalapb.UnknownFieldSet) = copy(unknownFields = __v)
      def discardUnknownFields = copy(unknownFields = _root_.scalapb.UnknownFieldSet.empty)
      def getFieldByNumber(__fieldNumber: _root_.scala.Int): _root_.scala.Any = {
        (__fieldNumber: @_root_.scala.unchecked) match {
          case 1 => {
            val __t = key
            if (__t != "") __t else null
          }
          case 2 => {
            val __t = value
            if (__t != "") __t else null
          }
        }
      }
      def getField(__field: _root_.scalapb.descriptors.FieldDescriptor): _root_.scalapb.descriptors.PValue = {
        _root_.scala.Predef.require(__field.containingMessage eq companion.scalaDescriptor)
        (__field.number: @_root_.scala.unchecked) match {
          case 1 => _root_.scalapb.descriptors.PString(key)
          case 2 => _root_.scalapb.descriptors.PString(value)
        }
      }
      def toProtoString: _root_.scala.Predef.String = _root_.scalapb.TextFormat.printToUnicodeString(this)
      def companion: snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.ResourceLabelsEntry.type = snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.ResourceLabelsEntry
      // @@protoc_insertion_point(GeneratedMessage[snapchat.research.gbml.SharedResourceConfig.ResourceLabelsEntry])
  }
  
  object ResourceLabelsEntry extends scalapb.GeneratedMessageCompanion[snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.ResourceLabelsEntry] {
    implicit def messageCompanion: scalapb.GeneratedMessageCompanion[snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.ResourceLabelsEntry] = this
    def parseFrom(`_input__`: _root_.com.google.protobuf.CodedInputStream): snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.ResourceLabelsEntry = {
      var __key: _root_.scala.Predef.String = ""
      var __value: _root_.scala.Predef.String = ""
      var `_unknownFields__`: _root_.scalapb.UnknownFieldSet.Builder = null
      var _done__ = false
      while (!_done__) {
        val _tag__ = _input__.readTag()
        _tag__ match {
          case 0 => _done__ = true
          case 10 =>
            __key = _input__.readStringRequireUtf8()
          case 18 =>
            __value = _input__.readStringRequireUtf8()
          case tag =>
            if (_unknownFields__ == null) {
              _unknownFields__ = new _root_.scalapb.UnknownFieldSet.Builder()
            }
            _unknownFields__.parseField(tag, _input__)
        }
      }
      snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.ResourceLabelsEntry(
          key = __key,
          value = __value,
          unknownFields = if (_unknownFields__ == null) _root_.scalapb.UnknownFieldSet.empty else _unknownFields__.result()
      )
    }
    implicit def messageReads: _root_.scalapb.descriptors.Reads[snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.ResourceLabelsEntry] = _root_.scalapb.descriptors.Reads{
      case _root_.scalapb.descriptors.PMessage(__fieldsMap) =>
        _root_.scala.Predef.require(__fieldsMap.keys.forall(_.containingMessage eq scalaDescriptor), "FieldDescriptor does not match message type.")
        snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.ResourceLabelsEntry(
          key = __fieldsMap.get(scalaDescriptor.findFieldByNumber(1).get).map(_.as[_root_.scala.Predef.String]).getOrElse(""),
          value = __fieldsMap.get(scalaDescriptor.findFieldByNumber(2).get).map(_.as[_root_.scala.Predef.String]).getOrElse("")
        )
      case _ => throw new RuntimeException("Expected PMessage")
    }
    def javaDescriptor: _root_.com.google.protobuf.Descriptors.Descriptor = snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.javaDescriptor.getNestedTypes().get(1)
    def scalaDescriptor: _root_.scalapb.descriptors.Descriptor = snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.scalaDescriptor.nestedMessages(1)
    def messageCompanionForFieldNumber(__number: _root_.scala.Int): _root_.scalapb.GeneratedMessageCompanion[_] = throw new MatchError(__number)
    lazy val nestedMessagesCompanions: Seq[_root_.scalapb.GeneratedMessageCompanion[_ <: _root_.scalapb.GeneratedMessage]] = Seq.empty
    def enumCompanionForFieldNumber(__fieldNumber: _root_.scala.Int): _root_.scalapb.GeneratedEnumCompanion[_] = throw new MatchError(__fieldNumber)
    lazy val defaultInstance = snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.ResourceLabelsEntry(
      key = "",
      value = ""
    )
    implicit class ResourceLabelsEntryLens[UpperPB](_l: _root_.scalapb.lenses.Lens[UpperPB, snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.ResourceLabelsEntry]) extends _root_.scalapb.lenses.ObjectLens[UpperPB, snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.ResourceLabelsEntry](_l) {
      def key: _root_.scalapb.lenses.Lens[UpperPB, _root_.scala.Predef.String] = field(_.key)((c_, f_) => c_.copy(key = f_))
      def value: _root_.scalapb.lenses.Lens[UpperPB, _root_.scala.Predef.String] = field(_.value)((c_, f_) => c_.copy(value = f_))
    }
    final val KEY_FIELD_NUMBER = 1
    final val VALUE_FIELD_NUMBER = 2
    @transient
    implicit val keyValueMapper: _root_.scalapb.TypeMapper[snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.ResourceLabelsEntry, (_root_.scala.Predef.String, _root_.scala.Predef.String)] =
      _root_.scalapb.TypeMapper[snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.ResourceLabelsEntry, (_root_.scala.Predef.String, _root_.scala.Predef.String)](__m => (__m.key, __m.value))(__p => snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.ResourceLabelsEntry(__p._1, __p._2))
    def of(
      key: _root_.scala.Predef.String,
      value: _root_.scala.Predef.String
    ): _root_.snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.ResourceLabelsEntry = _root_.snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.ResourceLabelsEntry(
      key,
      value
    )
    // @@protoc_insertion_point(GeneratedMessageCompanion[snapchat.research.gbml.SharedResourceConfig.ResourceLabelsEntry])
  }
  
  implicit class SharedResourceConfigLens[UpperPB](_l: _root_.scalapb.lenses.Lens[UpperPB, snapchat.research.gbml.gigl_resource_config.SharedResourceConfig]) extends _root_.scalapb.lenses.ObjectLens[UpperPB, snapchat.research.gbml.gigl_resource_config.SharedResourceConfig](_l) {
    def resourceLabels: _root_.scalapb.lenses.Lens[UpperPB, _root_.scala.collection.immutable.Map[_root_.scala.Predef.String, _root_.scala.Predef.String]] = field(_.resourceLabels)((c_, f_) => c_.copy(resourceLabels = f_))
    def commonComputeConfig: _root_.scalapb.lenses.Lens[UpperPB, snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.CommonComputeConfig] = field(_.getCommonComputeConfig)((c_, f_) => c_.copy(commonComputeConfig = Option(f_)))
    def optionalCommonComputeConfig: _root_.scalapb.lenses.Lens[UpperPB, _root_.scala.Option[snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.CommonComputeConfig]] = field(_.commonComputeConfig)((c_, f_) => c_.copy(commonComputeConfig = f_))
  }
  final val RESOURCE_LABELS_FIELD_NUMBER = 1
  final val COMMON_COMPUTE_CONFIG_FIELD_NUMBER = 2
  @transient
  private[gigl_resource_config] val _typemapper_resourceLabels: _root_.scalapb.TypeMapper[snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.ResourceLabelsEntry, (_root_.scala.Predef.String, _root_.scala.Predef.String)] = implicitly[_root_.scalapb.TypeMapper[snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.ResourceLabelsEntry, (_root_.scala.Predef.String, _root_.scala.Predef.String)]]
  def of(
    resourceLabels: _root_.scala.collection.immutable.Map[_root_.scala.Predef.String, _root_.scala.Predef.String],
    commonComputeConfig: _root_.scala.Option[snapchat.research.gbml.gigl_resource_config.SharedResourceConfig.CommonComputeConfig]
  ): _root_.snapchat.research.gbml.gigl_resource_config.SharedResourceConfig = _root_.snapchat.research.gbml.gigl_resource_config.SharedResourceConfig(
    resourceLabels,
    commonComputeConfig
  )
  // @@protoc_insertion_point(GeneratedMessageCompanion[snapchat.research.gbml.SharedResourceConfig])
}
