#!/usr/bin/env python3
"""
Script to extract training data from a raster GeoTiff file using a point or polygon Shapefile. The attributes from
the shapefile are copied to the output csv file. Pixels in overlapping geometries are not checked for repetition.
The output csv will contain all attributes in the sample file, in addition to band values and pixel center wkt geometry.

Script author: Richard Massey
Script usage: raster_sample_extract.py [-h] [--buffer BUFFER]  [--reducer REDUCER] [--tile_size TILE_SIZE]
              raster_file sample_shpfile output_csv

positional arguments:
  raster_file           GeoTiff raster file
  sample_shpfile        Shapefile containing point or polygon geometry type
  output_csv            Output csv file name

optional arguments:
  -h, --help            show this help message and exit
  --buffer BUFFER, -b BUFFER
                        Distance in projection coords to buffer the geometry (default: 0)
  --reducer REDUCER, -r REDUCER
                        Reducer to use for aggregating pixel values. Options: mean, median, min, max, std_dev, pctl_xx
                        (default: None)
  --tile_size TILE_SIZE, -t TILE_SIZE
                        Side of a square grid tile. Adjust for memory issues with large tiles.(default: 1024)
"""

from osgeo import gdal, ogr, gdal_array
import numpy as np
import datetime
import argparse
import logging
import sys
import os


class Logger(object):
    """
    Class for logging output.
    """
    def __init__(self, filename, name='default', level='info'):
        """
        Instantiate logger class object
        :param filename: Name of log file
        :param name: Name of logger
        :param level: Level of logging (options: critical, error, warning, info, debug - default:info)
        """
        self.logger = logging.getLogger(name)
        self.level = getattr(logging, level.upper())
        self.logger.setLevel(self.level)

        # create logging format
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')

        file_handler = logging.FileHandler(filename, mode='w')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def lprint(self, msg_str=''):
        """
        Method to print a string to logger
        :param msg_str: Message to print to log file
        """
        self.logger.log(self.level, msg_str)

    def close(self):
        """
        Close logger
        """
        file_handler = self.logger.handlers[0]
        file_handler.flush()
        file_handler.close()
        self.logger.removeHandler(file_handler)


class ImageProcessingError(Exception):
    status_code = 500

    def __init__(self, error="Image Processing Error",
                 description="Something went wrong processing your image",
                 status_code=status_code,
                 headers=None):
        """
        Class for errors in Image processing
        :param error: name of error
        :param description: readable description
        :param status_code: the http status code
        :param headers: any applicable headers
        :return:
        """
        self.description = description
        self.status_code = status_code
        self.headers = headers
        self.error = error


class Parser(object):
    """
    Parser object for Raster sample extraction
    """
    def __init__(self):
        description = "Script to extract training samples from a raster GeoTiff file " + \
                      "using a point or polygon Shapefile. The attributes from the shapefile are " + \
                      "copied to the output csv file. Pixels in overlapping geometries are " + \
                      "not checked for repetition."
        self.parser = argparse.ArgumentParser(description=description)

        # positional arguments
        self.parser.add_argument("raster_file", type=str, help="GeoTiff raster file")
        self.parser.add_argument("sample_shpfile", type=str, help="Shapefile containing point or polygon geometry type")
        self.parser.add_argument("output_csv", type=str, help="Output csv file name")

        # optional arguments
        self.parser.add_argument("--buffer", "-b", default=0, type=float,
                                 help="Distance in projection coords to buffer the geometry (default: 0)")
        self.parser.add_argument("--reducer", "-r", default=None, type=str,
                                 help="Reducer to use for aggregating pixel values. "
                                      "Options: mean, median, min, max, std_dev, pctl_xx (default: mean)")
        self.parser.add_argument("--tile_size", "-t", default=1024, type=int,
                                 help="Side of a square grid tile. Adjust for memory issues with large tiles."
                                      "(default: 1024)")


class Raster(object):
    """
    Class to read and write rasters from/to files and numpy arrays
    """

    def __init__(self, name, offsets=None, get_array=False, nan_replacement=0.0):
        """
        Raster class instantiate
        :param name: Raster file path
        :param offsets: Offsets for reading Raster array in array coordinate system
                        (pixel x coordinate, pixel y coordinate, x offset, y offset)
        :param get_array: If raster array should be read in memory
        :param nan_replacement: NAN or INF values, if found, will be replaced by this number
        """
        self.name = name
        self.bnames = []
        self.tile_grid = None

        # open raster file
        if os.path.isfile(self.name):
            fileptr = gdal.Open(self.name)  # open file
            self.datasource = fileptr
        else:
            raise ImageProcessingError('No datasource found')

        band_order = list(range(fileptr.RasterCount))

        # raster properties
        self.shape = [fileptr.RasterCount, fileptr.RasterYSize, fileptr.RasterXSize]
        self.dtype = fileptr.GetRasterBand(1).DataType
        self.nodatavalue = fileptr.GetRasterBand(1).GetNoDataValue()
        self.transform = fileptr.GetGeoTransform()
        self.crs_string = fileptr.GetProjection()
        self.bnames = list(fileptr.GetRasterBand(band_indx + 1).GetDescription()
                           for band_indx in range(self.shape[0]))

        if offsets is not None and len(offsets) == 4:
            self.offsets = offsets
        else:
            self.offsets = (0, 0, self.shape[2], self.shape[1])

        if get_array:
            sys.stdout.write('Reading bands: {}\n'.format(" ".join([self.bnames[band_indx]
                                                                    for band_indx in band_order])))

            self.array = np.zeros((self.shape[0], self.offsets[3], self.offsets[2]),
                                  gdal_array.GDALTypeCodeToNumericTypeCode(self.dtype))

            # read array and store the band values and name in array
            for band_indx, order_indx in enumerate(band_order):
                self.array[order_indx, :, :] = fileptr.GetRasterBand(band_indx + 1).ReadAsArray(*self.offsets)

            # if flag for finite values is present
            if np.isnan(self.array).any() or np.isinf(self.array).any():
                self.array[np.isnan(self.array)] = nan_replacement
                self.array[np.isinf(self.array)] = nan_replacement

        tie_pt = [self.transform[0], self.transform[3]]

        # raster bounds tuple of format: (xmin, xmax, ymin, ymax)
        self.bounds = [tie_pt[0], tie_pt[0] + self.transform[1] * self.shape[2],   # xmin, xmax
                       tie_pt[1] + self.transform[5] * self.shape[1], tie_pt[1]]   # ymin, ymax

    def __repr__(self):
        return "<Raster {} of size {}x{}x{} ".format(self.name, *self.shape)

    def make_tile_grid(self, tile_xsize=1024, tile_ysize=1024):
        """
        Returns the coordinates of square tile grid used to divide the raster into
        :param tile_xsize: Number of columns in the tile block (default: 1024)
        :param tile_ysize: Number of rows in the tile block (default: 1024)
        :return: dict of format {'block_coords': (x, y, cols, rows),
                                 'tie_point': (tie point x, tie point y),
                                 'tile_bounds': list of bounding polygon coordinates}
        """
        self.tile_grid = []

        xmin, xmax, ymin, ymax = (0, self.shape[2], 0, self.shape[1])

        for y in range(ymin, ymax, tile_ysize):
            rows = (y + tile_ysize < ymax) * tile_ysize + (y + tile_ysize >= ymax) * (ymax - y)

            for x in range(xmin, xmax, tile_xsize):
                cols = (x + tile_xsize < xmax) * tile_xsize + (x + tile_xsize >= xmax) * (xmax - x)

                tie_pt = (np.array([x, y]) * np.array([self.transform[1], self.transform[5]]) +
                          np.array([self.transform[0], self.transform[3]])).tolist()

                bounds = [tie_pt,
                          [tie_pt[0] + self.transform[1] * cols, tie_pt[1]],
                          [tie_pt[0] + self.transform[1] * cols, tie_pt[1] + self.transform[5] * rows],
                          [tie_pt[0], tie_pt[1] + self.transform[5] * rows],
                          tie_pt]

                self.tile_grid.append({'block_coords': (x, y, cols, rows),
                                       'tie_point': tie_pt,
                                       'tile_bounds': bounds})

    def get_tile(self, block_coords, nan_replacement=0.0):
        """
        Method to get raster numpy array of a tile
        :param block_coords: coordinates of tile to retrieve in image array coordinates
                             format is (upper left x, upper left y, tile columns, tile rows)
        :param nan_replacement: NAN or INF values, if found, will be replaced by this number
        :return: numpy array
        """
        tile_arr = np.zeros((self.shape[0], block_coords[3], block_coords[2]),
                            gdal_array.GDALTypeCodeToNumericTypeCode(self.dtype))

        for band_indx in range(self.shape[0]):
            temp_band = self.datasource.GetRasterBand(band_indx + 1)
            tile_arr[band_indx, :, :] = temp_band.ReadAsArray(*block_coords)

        if np.isnan(tile_arr).any() or np.isinf(tile_arr).any():
            tile_arr[np.isnan(tile_arr)] = nan_replacement
            tile_arr[np.isinf(tile_arr)] = nan_replacement

        return tile_arr

    def create_empty_dataset(self, tile):
        """
        Method to create an empty dataset with 1 band from tile properties
        :param tile: Tile dictionary from Raster.tile_grid list
        :return: GDAL SWIG dataset object
        """
        rows, cols = tile['block_coords'][2:4]
        tie_pt = tile['tie_point']
        dataset = gdal.GetDriverByName('MEM').Create('', cols, rows, 1, gdal.GDT_Byte)
        dataset.SetGeoTransform([tie_pt[0], self.transform[1], self.transform[2],
                                 tie_pt[1], self.transform[4], self.transform[5]])
        dataset.SetProjection(self.crs_string)
        band = dataset.GetRasterBand(1)
        band.Fill(0)
        return dataset

    @staticmethod
    def burn_to_raster(raster_dataset, vector_datasource, options=None, burn_value=1):
        """
        Method to burn a vector to a raster, creating a mask consisting of 0 and 1 values
        :param raster_dataset: GDAL SWIG raster dataset object
        :param vector_datasource: OGR SWIG vector object
        :param options: Options for burning values to raster
        :param burn_value: Values to burn to raster
        :return: raster array after burning
        """
        if options is None:
            options = ['ALL_TOUCHED=TRUE']
        result = gdal.RasterizeLayer(raster_dataset, [1], vector_datasource.GetLayer(),
                                     None, None, [burn_value], options)
        return raster_dataset.ReadAsArray()


class Vector(object):
    """
    Class for vector objects
    """
    def __init__(self, filename, buffer=0.0, split=False):
        """
        Constructor for class Vector
        :param filename: Name of the vector file (shapefile) with full path
        """
        self.filename = filename
        self.datasource = ogr.Open(self.filename)
        self.layer = self.datasource.GetLayerByIndex(0)
        self.spref = self.layer.GetSpatialRef()
        self.spref_str = self.spref.ExportToWkt()
        self.bounds = self.layer.GetExtent()
        self.type = self.layer.GetGeomType()
        self.name = self.layer.GetName()
        self.nfeat = self.layer.GetFeatureCount()

        self.split = split and (self.type in (4, 5, 6))

        self.attributes = []
        self.wkt_list = []
        self.geom_bounds = []

        feat = self.layer.GetNextFeature()
        while feat:
            all_items = feat.items()
            geom = feat.GetGeometryRef()

            if self.split:
                ngeom = geom.GetGeometryCount()
                for geom_indx in range(ngeom):
                    split_geom = geom.GetGeometryRef(geom_indx)
                    if buffer != 0.0:
                        split_geom = split_geom.Buffer(buffer)
                    split_bounds = split_geom.GetEnvelope()

                    self.attributes.append(all_items)
                    self.wkt_list.append(split_geom.ExportToWkt())
                    self.geom_bounds.append(split_bounds)
            else:
                if buffer != 0.0:
                    geom = geom.Buffer(buffer)
                bounds = geom.GetEnvelope()

                self.attributes.append(all_items)
                self.wkt_list.append(geom.ExportToWkt())
                self.geom_bounds.append(bounds)

            feat = self.layer.GetNextFeature()

    def __repr__(self):
        return '<Vector {} with {} elements>'.format(self.name, str(self.nfeat))

    def create_datasource(self, wkt_list, fid=0):
        """
        Method to create a vector data source from a list of geometries
        :param wkt_list: List of geometries in WKT string  format
        :param fid: Feature id
        :return: OGR SWIG datasource object
        """
        if type(wkt_list) not in (tuple, list):
            wkt_list = [wkt_list]

        datasource = ogr.GetDriverByName('Memory').CreateDataSource('')
        layer = datasource.CreateLayer('default', self.spref, self.type)
        fielddefn = ogr.FieldDefn('fid', ogr.OFTInteger)
        result = layer.CreateField(fielddefn)
        definition = layer.GetLayerDefn()

        for wkt in wkt_list:
            feat = ogr.Feature(definition)
            feat.SetGeometryDirectly(ogr.CreateGeometryFromWkt(wkt))
            feat.SetField('fid', fid)
            layer.CreateFeature(feat)
            feat = None
        return datasource


def array_reduce(array, method='mean', axis=0):
    """
    Method to reduce a 2D or 3D array using a specific method
    :param array: Numpy array
    :param method: Method to use for reduction (options: mean, median, std_dev, pctl_xx, min, max, default:mean
                                                         here xx is percentile)
    :param axis: Axis to apply the reducer along
    :return: float
    """
    methods = {'mean': np.mean,  'median': np.median, 'std_dev': np.std,
               'min': np.min, 'max': np.max, 'pctl': np.percentile}

    if 'pctl' in method:
        return methods['pctl'](array, int(method.split('_')[1]), axis=axis)
    elif method in ('mean', 'median', 'std_dev', 'min', 'max'):
        return methods[method](array, axis=axis)
    else:
        raise RuntimeError("Reducer {} is not implemented".format(method))


def write_to_csv(list_of_dicts, outfile=None):
    """
    Write list of dictionaries to CSV file
    :param list_of_dicts: List of dictionaries
    :param outfile: Output file name
    """
    if outfile is None:
        raise ValueError("No file name for writing")
    delimiter = ','
    if type(list_of_dicts) not in (tuple, list):
        list_of_dicts = [list_of_dicts]

    lines = list()
    lines.append(delimiter.join(list_of_dicts[0].keys()))
    for data_dict in list_of_dicts:
        lines.append(delimiter.join(list(str(val) for _, val in data_dict.items())))

    with open(outfile, 'w') as f:
        for line in lines:
            f.write(line + '\n')


def main(raster_filename, sample_shpfile, output_csv, logger, buffer=0.0, reducer=None, tile_size=1024):
    """
    Main method for sample extraction workflow
    :param raster_filename: Full filepath of raster geotiff file
    :param sample_shpfile: Full filepath of vector shapefile
    :param output_csv: Output CSV filepath
    :param logger: Logger object
    :param buffer: Buffer to be drawn around each geometry
    :param reducer: Method used to reduce multiple pixel values extracted using the same geometry
    :param tile_size: Raster processing unit size
    :return: None
    """
    vector = Vector(sample_shpfile, buffer, split=True)

    raster = Raster(raster_filename)
    raster.make_tile_grid(tile_size, tile_size)
    pixel_size = np.array([raster.transform[1], raster.transform[5]])

    out_geom_extract = list()
    for tile_indx, tile in enumerate(raster.tile_grid):

        logger.lprint('------- Processing tile {} of {} -------'.format(str(tile_indx + 1), str(len(raster.tile_grid))))
        tile_geom = ogr.CreateGeometryFromWkt('POLYGON(({}))'.format(', '.join(list(' '.join([str(x), str(y)])
                                                                                    for (x, y)
                                                                                    in tile['tile_bounds']))))
        tile_arr = raster.get_tile(block_coords=tile['block_coords'])
        tile_sample_list = list()

        # check if any geometry intersects tile
        for geom_indx, geom_wkt in enumerate(vector.wkt_list):
            samp_geom = ogr.CreateGeometryFromWkt(geom_wkt)

            if tile_geom.Intersects(samp_geom):
                tile_sample_list.append(geom_indx)

        if len(tile_sample_list) > 0:
            for list_indx, geom_indx in enumerate(tile_sample_list):

                logger.lprint('Geometry {} of {}'.format(str(list_indx + 1), str(len(tile_sample_list))))

                # burn the geometry on a blank raster to create a mask
                target_ds = raster.create_empty_dataset(tile)
                burn_datasource = vector.create_datasource(vector.wkt_list[geom_indx], fid=geom_indx)
                burnt_array = raster.burn_to_raster(target_ds, burn_datasource, burn_value=1)

                # get list of pixels
                pixel_xy_loc = np.vstack(np.where(burnt_array == 1)).T
                pixel_coord_array = pixel_xy_loc * pixel_size + np.array(tile['tie_point']) + pixel_size/2.0

                # create list of geometries for pixels
                pixel_wkt_list = list('POINT({})'.format(' '.join([str(coord) for coord in pixel_coords]))
                                      for pixel_coords in pixel_coord_array.tolist())

                # get attributes and band values
                attributes = vector.attributes[geom_indx]
                band_values = tile_arr[:, pixel_xy_loc[:, 0], pixel_xy_loc[:, 1]].T

                if reducer is not None:
                    reduced_values = array_reduce(band_values, method=reducer, axis=0)
                    out_dict = dict(zip(raster.bnames, reduced_values))
                    out_dict.update(attributes)

                    out_geom_extract.append(out_dict)

                else:
                    out_dict_list = list(dict(zip(raster.bnames, values)) for values in
                                         band_values.tolist())
                    for pixel_indx, out_dict in enumerate(out_dict_list):
                        out_dict.update(attributes)
                        out_dict.update({'geometry': pixel_wkt_list[pixel_indx]})

                    out_geom_extract += out_dict_list

                target_ds = burn_datasource = None

    logger.lprint('------ Completed extraction -----')

    if len(out_geom_extract) > 0:
        logger.lprint('Writing output file: {}'.format(output_csv))
        write_to_csv(out_geom_extract, outfile=output_csv)
    else:
        logger.lprint('No Samples found!')


if __name__ == "__main__":

    args = Parser().parser.parse_args()

    timestamp = datetime.datetime.now().isoformat().replace('-', '').replace(':', '').split('.')[0]
    logfile, _ = os.path.splitext(args.output_csv)
    logfile += '.'.join([timestamp, 'log'])
    output_logger = Logger(logfile)

    output_logger.lprint('Raster file: {}'.format(args.raster_file))
    output_logger.lprint('Sample file: {}'.format(args.sample_shpfile))
    output_logger.lprint('Output csv: {}'.format(args.output_csv))
    output_logger.lprint('Sample buffer: {}'.format(args.buffer))

    main(args.raster_file, args.sample_shpfile, args.output_csv, output_logger,
         args.buffer, args.reducer, args.tile_size)

    output_logger.lprint('---------------------------------------------- Done!')
    output_logger.close()
