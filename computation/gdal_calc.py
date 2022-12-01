import os
import os.path
import string
from collections import defaultdict
from numbers import Number
from typing import Union, Tuple, Optional, Sequence, Dict

import numpy
from osgeo import gdal
from osgeo import gdal_array
from osgeo_utils.auxiliary import extent_util
from osgeo_utils.auxiliary.base import is_path_like, PathLikeOrStr, MaybeSequence
from osgeo_utils.auxiliary.color_table import get_color_table, ColorTableLike
from osgeo_utils.auxiliary.extent_util import Extent, GT
from osgeo_utils.auxiliary.rectangle import GeoRectangle
from osgeo_utils.auxiliary.util import GetOutputDriverFor, open_ds

GDALDataType = int

# create alphabetic list (lowercase + uppercase) for storing input layers
AlphaList = list(string.ascii_letters)

# set up some default nodatavalues for each datatype
DefaultNDVLookup = {gdal.GDT_Byte: 255, gdal.GDT_UInt16: 65535, gdal.GDT_Int16: -32768,
                    gdal.GDT_UInt32: 4294967293, gdal.GDT_Int32: -2147483647,
                    gdal.GDT_Float32: 3.402823466E+38, gdal.GDT_Float64: 1.7976931348623158E+308}

# tuple of available output datatypes names
GDALDataTypeNames = tuple(gdal.GetDataTypeName(dt) for dt in DefaultNDVLookup.keys())

""" Perform raster calculations with numpy syntax.
Use any basic arithmetic supported by numpy arrays such as +-* along with logical
operators such as >. Note that all files must have the same dimensions, but no projection checking is performed.
Keyword arguments:
    [A-Z]: input files
    [A_band - Z_band]: band to use for respective input file
Examples:
add two files together:
    Calc("A+B", A="input1.tif", B="input2.tif", outfile="result.tif")
average of two layers:
    Calc(calc="(A+B)/2", A="input1.tif", B="input2.tif", outfile="result.tif")
set values of zero and below to null:
    Calc(calc="A*(A>0)", A="input.tif", A_band=2, outfile="result.tif", NoDataValue=0)
work with two bands:
    Calc(["(A+B)/2", "A*(A>0)"], A="input.tif", A_band=1, B="input.tif", B_band=2, outfile="result.tif", NoDataValue=0)
sum all files with hidden noDataValue
    Calc(calc="sum(a,axis=0)", a=['0.tif','1.tif','2.tif'], outfile="sum.tif", hideNoData=True)
"""

def Calc(
        calc: MaybeSequence[str],
        outfile: Optional[PathLikeOrStr] = None,
        NoDataValue: Optional[Number] = None,
        type: Optional[Union[GDALDataType, str]] = None,
        format: Optional[str] = None,
        creation_options: Optional[Sequence[str]] = None,
        allBands: str = '',
        overwrite: bool = False,
        hideNoData: bool = False,
        projectionCheck: bool = False,
        color_table: Optional[ColorTableLike] = None,
        extent: Optional[Extent] = None,
        projwin: Optional[Union[Tuple, GeoRectangle]] = None,
        user_namespace: Optional[Dict]=None,
        debug: bool = False,
        quiet: bool = False,
        **input_files):

    if debug:
        print(f"gdal_calc.py starting calculation {calc}")

    # Single calc value compatibility
    if isinstance(calc, (list, tuple)):
        calc = calc
    else:
        calc = [calc]
    calc = [c.strip('"') for c in calc]

    creation_options = creation_options or []

    # set up global namespace for eval with all functions of gdal_array, numpy
    global_namespace = {key: getattr(module, key)
                        for module in [gdal_array, numpy] for key in dir(module) if not key.startswith('__')}

    if user_namespace:
        global_namespace.update(user_namespace)

    if not calc:
        raise Exception("No calculation provided.")
    elif not outfile and format.upper() != 'MEM':
        raise Exception("No output file provided.")

    if format is None:
        format = GetOutputDriverFor(outfile)

    if isinstance(extent, GeoRectangle):
        pass
    elif projwin:
        if isinstance(projwin, GeoRectangle):
            extent = projwin
        else:
            extent = GeoRectangle.from_lurd(*projwin)
    elif not extent:
        extent = Extent.IGNORE
    else:
        extent = extent_util.parse_extent(extent)

    compatible_gt_eps = 0.000001
    gt_diff_support = {
        GT.INCOMPATIBLE_OFFSET: extent != Extent.FAIL,
        GT.INCOMPATIBLE_PIXEL_SIZE: False,
        GT.INCOMPATIBLE_ROTATION: False,
        GT.NON_ZERO_ROTATION: False,
    }
    gt_diff_error = {
        GT.INCOMPATIBLE_OFFSET: 'different offset',
        GT.INCOMPATIBLE_PIXEL_SIZE: 'different pixel size',
        GT.INCOMPATIBLE_ROTATION: 'different rotation',
        GT.NON_ZERO_ROTATION: 'non zero rotation',
    }

    ################################################################
    # fetch details of input layers
    ################################################################

    # set up some lists to store data for each band
    myFileNames = []  # input filenames
    myFiles = []  # input DataSets
    myBands = []  # input bands
    myAlphaList = []  # input alpha letter that represents each input file
    myDataType = []  # string representation of the datatype of each input file
    myDataTypeNum = []  # datatype of each input file
    myNDV = []  # nodatavalue for each input file
    DimensionsCheck = None  # dimensions of the output
    Dimensions = []  # Dimensions of input files
    ProjectionCheck = None  # projection of the output
    GeoTransformCheck = None  # GeoTransform of the output
    GeoTransforms = []  # GeoTransform of each input file
    GeoTransformDiffer = False  # True if we have inputs with different GeoTransforms
    myTempFileNames = []  # vrt filename from each input file
    myAlphaFileLists = []  # list of the Alphas which holds a list of inputs

    # loop through input files - checking dimensions
    for alphas, filenames in input_files.items():
        if isinstance(filenames, (list, tuple)):
            # alpha is a list of files
            myAlphaFileLists.append(alphas)
        elif is_path_like(filenames) or isinstance(filenames, gdal.Dataset):
            # alpha is a single filename or a Dataset
            filenames = [filenames]
            alphas = [alphas]
        else:
            # I guess this alphas should be in the global_namespace,
            # It would have been better to pass it as user_namepsace, but I'll accept it anyway
            global_namespace[alphas] = filenames
            continue
        for alpha, filename in zip(alphas * len(filenames), filenames):
            if not alpha.endswith("_band"):
                # check if we have asked for a specific band...
                alpha_band = f"{alpha}_band"
                if alpha_band in input_files:
                    myBand = input_files[alpha_band]
                else:
                    myBand = 1

                myF_is_ds = not is_path_like(filename)
                if myF_is_ds:
                    myFile = filename
                    filename = None
                else:
                    myFile = open_ds(filename, gdal.GA_ReadOnly)
                if not myFile:
                    raise IOError(f"No such file or directory: '{filename}'")

                myFileNames.append(filename)
                myFiles.append(myFile)
                myBands.append(myBand)
                myAlphaList.append(alpha)
                dt = myFile.GetRasterBand(myBand).DataType
                myDataType.append(gdal.GetDataTypeName(dt))
                myDataTypeNum.append(dt)
                myNDV.append(None if hideNoData else myFile.GetRasterBand(myBand).GetNoDataValue())

                # check that the dimensions of each layer are the same
                myFileDimensions = [myFile.RasterXSize, myFile.RasterYSize]
                if DimensionsCheck:
                    if DimensionsCheck != myFileDimensions:
                        GeoTransformDiffer = True
                        if extent in [Extent.IGNORE, Extent.FAIL]:
                            raise Exception(
                                f"Error! Dimensions of file {filename} ({myFileDimensions[0]:d}, "
                                f"{myFileDimensions[1]:d}) are different from other files "
                                f"({DimensionsCheck[0]:d}, {DimensionsCheck[1]:d}).  Cannot proceed")
                else:
                    DimensionsCheck = myFileDimensions

                # check that the Projection of each layer are the same
                myProjection = myFile.GetProjection()
                if ProjectionCheck:
                    if projectionCheck and ProjectionCheck != myProjection:
                        raise Exception(
                            f"Error! Projection of file {filename} {myProjection} "
                            f"are different from other files {ProjectionCheck}.  Cannot proceed")
                else:
                    ProjectionCheck = myProjection

                # check that the GeoTransforms of each layer are the same
                myFileGeoTransform = myFile.GetGeoTransform(can_return_null=True)
                if extent == Extent.IGNORE:
                    GeoTransformCheck = myFileGeoTransform
                else:
                    Dimensions.append(myFileDimensions)
                    GeoTransforms.append(myFileGeoTransform)
                    if not GeoTransformCheck:
                        GeoTransformCheck = myFileGeoTransform
                    else:
                        my_gt_diff = extent_util.gt_diff(GeoTransformCheck, myFileGeoTransform, eps=compatible_gt_eps,
                                                         diff_support=gt_diff_support)
                        if my_gt_diff not in [GT.SAME, GT.ALMOST_SAME]:
                            GeoTransformDiffer = True
                            if my_gt_diff != GT.COMPATIBLE_DIFF:
                                raise Exception(
                                    f"Error! GeoTransform of file {filename} {myFileGeoTransform} is incompatible "
                                    f"({gt_diff_error[my_gt_diff]}), first file GeoTransform is {GeoTransformCheck}. "
                                    f"Cannot proceed")
                if debug:
                    print(
                        f"file {alpha}: {filename}, dimensions: "
                        f"{DimensionsCheck[0]}, {DimensionsCheck[1]}, type: {myDataType[-1]}")

    # process allBands option
    allBandsIndex = None
    allBandsCount = 1
    if allBands:
        if len(calc) > 1:
            raise Exception("Error! --allBands implies a single --calc")
        try:
            allBandsIndex = myAlphaList.index(allBands)
        except ValueError:
            raise Exception(f"Error! allBands option was given but Band {allBands} not found.  Cannot proceed")
        allBandsCount = myFiles[allBandsIndex].RasterCount
        if allBandsCount <= 1:
            allBandsIndex = None
    else:
        allBandsCount = len(calc)

    if extent not in [Extent.IGNORE, Extent.FAIL] and (
        GeoTransformDiffer or isinstance(extent, GeoRectangle)):
        # mixing different GeoTransforms/Extents
        GeoTransformCheck, DimensionsCheck, ExtentCheck = extent_util.calc_geotransform_and_dimensions(
            GeoTransforms, Dimensions, extent)
        if GeoTransformCheck is None:
            raise Exception("Error! The requested extent is empty. Cannot proceed")
        for i in range(len(myFileNames)):
            temp_vrt_filename, temp_vrt_ds = extent_util.make_temp_vrt(myFiles[i], ExtentCheck)
            myTempFileNames.append(temp_vrt_filename)
            myFiles[i] = None  # close original ds
            myFiles[i] = temp_vrt_ds  # replace original ds with vrt_ds

            # update the new precise dimensions and gt from the new ds
            GeoTransformCheck = temp_vrt_ds.GetGeoTransform()
            DimensionsCheck = [temp_vrt_ds.RasterXSize, temp_vrt_ds.RasterYSize]
        temp_vrt_ds = None

    ################################################################
    # set up output file
    ################################################################

    # open output file exists
    if outfile and os.path.isfile(outfile) and not overwrite:
        if allBandsIndex is not None:
            raise Exception("Error! allBands option was given but Output file exists, must use --overwrite option!")
        if len(calc) > 1:
            raise Exception(
                "Error! multiple calc options were given but Output file exists, must use --overwrite option!")
        if debug:
            print(f"Output file {outfile} exists - filling in results into file")

        myOut = open_ds(outfile, gdal.GA_Update)
        if myOut is None:
            error = 'but cannot be opened for update'
        elif [myOut.RasterXSize, myOut.RasterYSize] != DimensionsCheck:
            error = 'but is the wrong size'
        elif ProjectionCheck and ProjectionCheck != myOut.GetProjection():
            error = 'but is the wrong projection'
        elif GeoTransformCheck and GeoTransformCheck != myOut.GetGeoTransform(can_return_null=True):
            error = 'but is the wrong geotransform'
        else:
            error = None
        if error:
            raise Exception(
                f"Error! Output exists, {error}.  Use the --overwrite option "
                f"to automatically overwrite the existing file")

        myOutB = myOut.GetRasterBand(1)
        myOutNDV = myOutB.GetNoDataValue()
        myOutType = myOutB.DataType

    else:
        if outfile:
            # remove existing file and regenerate
            if os.path.isfile(outfile):
                os.remove(outfile)
            # create a new file
            if debug:
                print(f"Generating output file {outfile}")
        else:
            outfile = ''

        # find data type to use
        if not type:
            # use the largest type of the input files
            myOutType = max(myDataTypeNum)
        else:
            myOutType = type
            if isinstance(myOutType, str):
                myOutType = gdal.GetDataTypeByName(myOutType)

        # create file
        myOutDrv = gdal.GetDriverByName(format)
        myOut = myOutDrv.Create(
            os.fspath(outfile), DimensionsCheck[0], DimensionsCheck[1], allBandsCount,
            myOutType, creation_options)

        # set output geo info based on first input layer
        if not GeoTransformCheck:
            GeoTransformCheck = myFiles[0].GetGeoTransform(can_return_null=True)
        if GeoTransformCheck:
            myOut.SetGeoTransform(GeoTransformCheck)

        if not ProjectionCheck:
            ProjectionCheck = myFiles[0].GetProjection()
        if ProjectionCheck:
            myOut.SetProjection(ProjectionCheck)

        if NoDataValue is None:
            myOutNDV = DefaultNDVLookup[myOutType]  # use the default noDataValue for this datatype
        elif isinstance(NoDataValue, str) and NoDataValue.lower() == 'none':
            myOutNDV = None  # not to set any noDataValue
        else:
            myOutNDV = NoDataValue  # use the given noDataValue

        for i in range(1, allBandsCount + 1):
            myOutB = myOut.GetRasterBand(i)
            if myOutNDV is not None:
                myOutB.SetNoDataValue(myOutNDV)
            if color_table:
                # set color table and color interpretation
                if is_path_like(color_table):
                    color_table = get_color_table(color_table)
                myOutB.SetRasterColorTable(color_table)
                myOutB.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)

            myOutB = None  # write to band

        if hideNoData:
            myOutNDV = None

    myOutTypeName = gdal.GetDataTypeName(myOutType)
    if debug:
        print(f"output file: {outfile}, dimensions: {myOut.RasterXSize}, {myOut.RasterYSize}, type: {myOutTypeName}")

    ################################################################
    # find block size to chop grids into bite-sized chunks
    ################################################################

    # use the block size of the first layer to read efficiently
    myBlockSize = myFiles[0].GetRasterBand(myBands[0]).GetBlockSize()
    # find total x and y blocks to be read
    nXBlocks = (int)((DimensionsCheck[0] + myBlockSize[0] - 1) / myBlockSize[0])
    nYBlocks = (int)((DimensionsCheck[1] + myBlockSize[1] - 1) / myBlockSize[1])
    myBufSize = myBlockSize[0] * myBlockSize[1]

    if debug:
        print(f"using blocksize {myBlockSize[0]} x {myBlockSize[1]}")

    # variables for displaying progress
    ProgressCt = -1
    ProgressMk = -1
    ProgressEnd = nXBlocks * nYBlocks * allBandsCount

    ################################################################
    # start looping through each band in allBandsCount
    ################################################################

    for bandNo in range(1, allBandsCount + 1):

        ################################################################
        # start looping through blocks of data
        ################################################################

        # store these numbers in variables that may change later
        nXValid = myBlockSize[0]
        nYValid = myBlockSize[1]

        # loop through X-lines
        for X in range(0, nXBlocks):

            # in case the blocks don't fit perfectly
            # change the block size of the final piece
            if X == nXBlocks - 1:
                nXValid = DimensionsCheck[0] - X * myBlockSize[0]

            # find X offset
            myX = X * myBlockSize[0]

            # reset buffer size for start of Y loop
            nYValid = myBlockSize[1]
            myBufSize = nXValid * nYValid

            # loop through Y lines
            for Y in range(0, nYBlocks):
                ProgressCt += 1
                if 10 * ProgressCt / ProgressEnd % 10 != ProgressMk and not quiet:
                    ProgressMk = 10 * ProgressCt / ProgressEnd % 10
                    from sys import version_info
                    if version_info >= (3, 0, 0):
                        exec('print("%d.." % (10*ProgressMk), end=" ")')
                    else:
                        exec('print 10*ProgressMk, "..",')

                # change the block size of the final piece
                if Y == nYBlocks - 1:
                    nYValid = DimensionsCheck[1] - Y * myBlockSize[1]
                    myBufSize = nXValid * nYValid

                # find Y offset
                myY = Y * myBlockSize[1]

                # create empty buffer to mark where nodata occurs
                myNDVs = None

                # make local namespace for calculation
                local_namespace = {}

                val_lists = defaultdict(list)

                # fetch data for each input layer
                for i, Alpha in enumerate(myAlphaList):

                    # populate lettered arrays with values
                    if allBandsIndex is not None and allBandsIndex == i:
                        myBandNo = bandNo
                    else:
                        myBandNo = myBands[i]
                    myval = gdal_array.BandReadAsArray(myFiles[i].GetRasterBand(myBandNo),
                                                       xoff=myX, yoff=myY,
                                                       win_xsize=nXValid, win_ysize=nYValid)
                    if myval is None:
                        raise Exception(f'Input block reading failed from filename {filename[i]}')

                    # fill in nodata values
                    if myNDV[i] is not None:
                        # myNDVs is a boolean buffer.
                        # a cell equals to 1 if there is NDV in any of the corresponding cells in input raster bands.
                        if myNDVs is None:
                            # this is the first band that has NDV set. we initializes myNDVs to a zero buffer
                            # as we didn't see any NDV value yet.
                            myNDVs = numpy.zeros(myBufSize)
                            myNDVs.shape = (nYValid, nXValid)
                        myNDVs = 1 * numpy.logical_or(myNDVs == 1, myval == myNDV[i])

                    # add an array of values for this block to the eval namespace
                    if Alpha in myAlphaFileLists:
                        val_lists[Alpha].append(myval)
                    else:
                        local_namespace[Alpha] = myval
                    myval = None

                for lst in myAlphaFileLists:
                    local_namespace[lst] = val_lists[lst]

                # try the calculation on the array blocks
                this_calc = calc[bandNo - 1 if len(calc) > 1 else 0]
                try:
                    myResult = eval(this_calc, global_namespace, local_namespace)
                except:
                    print(f"evaluation of calculation {this_calc} failed")
                    raise

                # Propagate nodata values (set nodata cells to zero
                # then add nodata value to these cells).
                if myNDVs is not None and myOutNDV is not None:
                    myResult = ((1 * (myNDVs == 0)) * myResult) + (myOutNDV * myNDVs)
                elif not isinstance(myResult, numpy.ndarray):
                    myResult = numpy.ones((nYValid, nXValid)) * myResult

                # write data block to the output file
                myOutB = myOut.GetRasterBand(bandNo)
                if gdal_array.BandWriteArray(myOutB, myResult, xoff=myX, yoff=myY) != 0:
                    raise Exception('Block writing failed')
                myOutB = None  # write to band

    # remove temp files
    for idx, tempFile in enumerate(myTempFileNames):
        myFiles[idx] = None
        os.remove(tempFile)

    gdal.ErrorReset()
    myOut.FlushCache()
    if gdal.GetLastErrorMsg() != '':
        raise Exception('Dataset writing failed')

    if not quiet:
        print("100 - Done")

    return myOut


def doit(opts):
    kwargs = vars(opts)
    if 'outF' in kwargs:
        kwargs["outfile"] = kwargs.pop('outF')
    return Calc(**kwargs)