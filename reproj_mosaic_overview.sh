#!/bin/bash
#SBATCH --job-name=tcproc
#SBATCH --time=3-23:59:59
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=204800

# SLURM script to download, reproject, mosaic, and create overview of raster geotiff files

date

year=2010
echo 'Begin Download!~~~~~~~~~~'

downloaddir='/scratch/user/gdrive/sync/gz'$year

if [ ! -d $downloaddir ] ; then
	mkdir $downloaddir;
fi

rclone --include "raster_*.tif" copy myRemote:/work/is/fine/ $downloaddir

echo 'End Download!~~~~~~~~~~'
echo '********************************************************************************************'
echo 'Begin Reprojection!~~~~~~~~~~'

datadir='/scratch/user/gdrive/sync/tif'$year'/'
outdir='/scratch/user/gdrive/sync/tif_geo'$year'/'

if [ ! -d $datadir ] ; then
	mkdir $datadir;
fi

if [ ! -d $outdir ] ; then
	mkdir $outdir;
fi

export GDAL_DATA='/home/user/path/gdal'

# list of input files
files=(${datadir}*.tif)
echo ${files[*]}

echo '********************************************************************************************'

# batch reproject geotiff files
for f in ${files[*]}; do
   bname=$(basename $f);
   extension="${bname##*.}";
   filename="${bname%.*}";
   outf=$outdir""$filename"_geo."$extension;
   if [ -f $outf ] ; then
      rm -f $outf;
   fi;
   echo 'Reprojecting '$f' to '$outf;
   gdalwarp -overwrite -multi $f $outf -et 0.05 -ot Byte -tr 0.00027 0.00027 -t_srs 'EPSG:4326' -wo WRITE_FLUSH=YES -wo NUM_THREADS=4 ;
done

echo '********************************************************************************************'
echo 'Begin Mosaic!~~~~~~~~~~'

mosaic=$outdir'mosaic.tif'
compmosaic=$outdir'mosaic_compressed.tif'

echo 'Data folder: '$datadir
echo 'Mosaic: '$mosaic
echo 'Compressed mosaic: '$compmosaic
echo '********************************************************************************************'

# mosaic geotiff files at spatial resolution: 30m
# data type: Byte, background value: 0, file format: Geotiff
gdal_merge.py -init 0 -o $mosaic -of GTiff -ps 0.00027 0.00027 -ot Byte ${files[*]}

# compress large tif file using LZW compression, use BIGTIFF=YES for large files
gdal_translate -of GTiff -co COMPRESS=LZW -co BIGTIFF=YES $mosaic $compmosaic

# make overview (pyramid) file: gdaladdo -> gdal add overview
# this is useful if at any point ArcGIS is going to be used with this data
# this makes pyramids and will save that step with ArcGIS
gdaladdo -ro $compmosaic 2 4 8 16 32 64 128 256 --config COMPRESS_OVERVIEW LZW

echo '********************************************************************************************'
date
