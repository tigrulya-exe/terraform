import os

import numpy as np
from osgeo import gdal
from osgeo_utils.auxiliary.util import open_ds, GetOutputDriverFor
from osgeo_utils.gdal_calc import DefaultNDVLookup


class RasterInfo:
    def __init__(self, uid, path, band=None):
        self.uid = uid
        self.path = path
        self.band = band or 1


# Rewrite of gdal raster calc, that accepts
# computation func instead of expression
class SimpleRasterCalc:
    def calculate(
            self,
            func,
            output_path: str,
            raster_infos: list[RasterInfo],
            hideNoData: bool = False,
            overwrite_out: bool = False,
            debug: bool = True,
            NoDataValue=None,
            **kwargs):
        datasets_by_path = {}
        for info in raster_infos:
            ds = datasets_by_path.get(info.path)
            if not ds:
                ds = open_ds(info.path)
                datasets_by_path[info.path] = ds

        format = GetOutputDriverFor(output_path)

        ################################################################
        # fetch details of input layers
        ################################################################

        # set up some lists to store data for each band
        myFileNames = []  # input filenames
        myFiles = []  # input DataSets
        myBands = []  # input bands
        myDataType = []  # string representation of the datatype of each input file
        myDataTypeNum = []  # datatype of each input file
        myNDV = []  # nodatavalue for each input file
        DimensionsCheck = None  # dimensions of the output
        ProjectionCheck = None  # projection of the output
        GeoTransformCheck = None  # GeoTransform of the output

        # loop through input files - checking dimensions
        for raster_info in raster_infos:
            # todo
            alpha, filename, myBand = raster_info.uid, raster_info.path, raster_info.band
            myFile = datasets_by_path[filename]

            myFileNames.append(filename)
            myFiles.append(myFile)
            myBands.append(myBand)
            dt = myFile.GetRasterBand(myBand).DataType
            myDataType.append(gdal.GetDataTypeName(dt))
            myDataTypeNum.append(dt)
            myNDV.append(None if hideNoData else myFile.GetRasterBand(myBand).GetNoDataValue())

            # check that the dimensions of each layer are the same
            myFileDimensions = [myFile.RasterXSize, myFile.RasterYSize]
            if DimensionsCheck:
                if DimensionsCheck != myFileDimensions:
                    raise Exception(
                        f"Error! Dimensions of file {filename} ({myFileDimensions[0]:d}, "
                        f"{myFileDimensions[1]:d}) are different from other files "
                        f"({DimensionsCheck[0]:d}, {DimensionsCheck[1]:d}).  Cannot proceed")
            else:
                DimensionsCheck = myFileDimensions

            # check that the Projection of each layer are the same
            myProjection = myFile.GetProjection()
            if ProjectionCheck:
                if ProjectionCheck != myProjection:
                    raise Exception(
                        f"Error! Projection of file {filename} {myProjection} "
                        f"are different from other files {ProjectionCheck}.  Cannot proceed")
            else:
                ProjectionCheck = myProjection

            # check that the GeoTransforms of each layer are the same
            myFileGeoTransform = myFile.GetGeoTransform(can_return_null=True)
            GeoTransformCheck = myFileGeoTransform
            # print(f"file {alpha}: {filename}, dimensions: "
            #       f"{DimensionsCheck[0]}, {DimensionsCheck[1]}, type: {myDataType[-1]}")

        allBandsCount = 1

        ################################################################
        # set up output file
        ################################################################

        # open output file exists
        if output_path and os.path.isfile(output_path) and not overwrite_out:
            myOut = open_ds(output_path, gdal.GA_Update)
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
            if output_path:
                # remove existing file and regenerate
                if os.path.isfile(output_path):
                    os.remove(output_path)
                # create a new file
                if debug:
                    print(f"Generating output file {output_path}")
            else:
                outfile = ''

            # find data type to use
            myOutType = max(myDataTypeNum)

            # create file
            myOutDrv = gdal.GetDriverByName(format)
            myOut = myOutDrv.Create(
                os.fspath(output_path),
                DimensionsCheck[0],
                DimensionsCheck[1],
                allBandsCount,
                myOutType)

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

            myOutB = myOut.GetRasterBand(1)
            if myOutNDV is not None:
                myOutB.SetNoDataValue(myOutNDV)

            if hideNoData:
                myOutNDV = None

        myOutTypeName = gdal.GetDataTypeName(myOutType)
        if debug:
            print(
                f"output file: {output_path}, dimensions: {myOut.RasterXSize}, {myOut.RasterYSize}, type: {myOutTypeName}")

        # out_format = GetOutputDriverFor(output_path)
        # out_driver = gdal.GetDriverByName(out_format)
        # out_ds = out_driver.Create(
        #     os.fspath(output_path),
        #     DimensionsCheck[0],
        #     DimensionsCheck[1],
        #     1,
        #     myOutType
        # )

        ################################################################
        # MY CALCULATIONS
        ################################################################

        first_info = raster_infos[0]
        first_ds = datasets_by_path[first_info.path]
        # use the block size of the first layer to read efficiently
        block_size = first_ds.GetRasterBand(first_info.band).GetBlockSize()
        # find total x and y blocks to be read
        x_blocks_count = int((first_ds.RasterXSize + block_size[0] - 1) / block_size[0])
        y_blocks_count = int((first_ds.RasterYSize + block_size[1] - 1) / block_size[1])

        x_block_size, y_block_size = block_size

        # loop through X-lines
        for x_block_idx in range(0, x_blocks_count):

            # in case the blocks don't fit perfectly
            # change the block size of the final piece
            if x_block_idx == x_blocks_count - 1:
                x_block_size = first_ds.RasterXSize - x_block_idx * block_size[0]

            # find X offset
            x_offset = x_block_idx * block_size[0]

            # reset buffer size for start of Y loop
            y_block_size = block_size[1]
            buf_size = x_block_size * y_block_size

            # loop through Y lines
            for y_block_idx in range(0, y_blocks_count):
                # change the block size of the final piece
                if y_block_idx == y_blocks_count - 1:
                    y_block_size = first_ds.RasterYSize - y_block_idx * block_size[1]
                    buf_size = x_block_size * y_block_size

                # find Y offset
                y_offset = y_block_idx * block_size[1]

                # create empty buffer to mark where nodata occurs
                myNDVs = None

                calc_func_kw_args = {}
                for i, raster_info in enumerate(raster_infos):
                    band = datasets_by_path[raster_info.path].GetRasterBand(raster_info.band)
                    block = band.ReadAsArray(
                        xoff=x_offset,
                        yoff=y_offset,
                        win_xsize=x_block_size,
                        win_ysize=y_block_size
                    )

                    # fill in nodata values
                    if myNDV[i] is not None:
                        # myNDVs is a boolean buffer.
                        # a cell equals to 1 if there is NDV in any of the corresponding cells in input raster bands.
                        if myNDVs is None:
                            # this is the first band that has NDV set. we initialize myNDVs to a zero buffer
                            # as we didn't see any NDV value yet.
                            myNDVs = np.zeros(buf_size)
                            myNDVs.shape = (y_block_size, x_block_size)
                        myNDVs = 1 * np.logical_or(myNDVs == 1, block == myNDV[i])

                    calc_func_kw_args[raster_info.uid] = block

                calc_res = func(**calc_func_kw_args, **kwargs)

                # Propagate nodata values (set nodata cells to zero
                # then add nodata value to these cells).
                if myNDVs is not None and myOutNDV is not None:
                    calc_res = ((1 * (myNDVs == 0)) * calc_res) + (myOutNDV * myNDVs)

                # write data block to the output file
                myOutB = myOut.GetRasterBand(1)
                if myOutB.WriteArray(calc_res, xoff=x_offset, yoff=y_offset) != 0:
                    raise Exception('Block writing failed')

        datasets_by_path = None
