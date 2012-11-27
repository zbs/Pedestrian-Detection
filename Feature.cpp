#include "Feature.h"

void
FeatureExtractor::operator()(const ImageDatabase& db, FeatureSet& featureSet) const
{
	int n = db.getSize();

	featureSet.resize(n);
	for(int i = 0; i < n; i++) {
		CByteImage img;
		ReadFile(img, db.getFilename(i).c_str());

		featureSet[i] = (*this)(img);
	}
}

CByteImage
FeatureExtractor::render(const Feature& f, bool normalizeFeat) const
{
	if(normalizeFeat) {
		CShape shape = f.Shape();
		Feature fAux(shape);

		float fMin, fMax;
		f.getRangeOfValues(fMin, fMax);

		for(int y = 0; y < shape.height; y++) {
			float* fIt = (float*) f.PixelAddress(0,y,0);
			float* fAuxIt = (float*) fAux.PixelAddress(0,y,0);

			for(int x = 0; x < shape.width * shape.nBands; x++, fAuxIt++, fIt++) {
				*fAuxIt = (*fIt) / fMax;
			}
		}

		return this->render(fAux);
	} else {
		return this->render(f);
	}
}

FeatureExtractor* 
FeatureExtractorNew(const char* featureType)
{
	if(strcasecmp(featureType, "tinyimg") == 0) return new TinyImageFeatureExtractor();
	if(strcasecmp(featureType, "hog") == 0) return new HOGFeatureExtractor();
	// Implement other features or call a feature extractor with a different set
	// of parameters by adding more calls here.
	if(strcasecmp(featureType, "myfeat1") == 0) throw CError("not implemented");
	if(strcasecmp(featureType, "myfeat2") == 0) throw CError("not implemented");
	if(strcasecmp(featureType, "myfeat3") == 0) throw CError("not implemented");
	else {
		throw CError("Unknown feature type: %s", featureType);
	}
}

// ============================================================================
// TinyImage
// ============================================================================

TinyImageFeatureExtractor::TinyImageFeatureExtractor(int targetWidth, int targetHeight):
_targetW(targetWidth), _targetH(targetHeight)
{
}

Feature 
TinyImageFeatureExtractor::operator()(const CByteImage& img_) const
{
	CFloatImage tinyImg(_targetW, _targetH, 1);

	CFloatImage floatImage;
	CFloatImage grayImage;
	TypeConvert(img_, floatImage);
	convertRGB2GrayImage(floatImage, grayImage);
	CTransform3x3 scaleXform = CTransform3x3::Scale(_targetW / img_.Shape().width, _targetH / img_.Shape().height);
	WarpGlobal(grayImage, tinyImg, scaleXform.Inverse(), eWarpInterpLinear, 1.0f);

	return tinyImg;
}

CByteImage 
TinyImageFeatureExtractor::render(const Feature& f) const
{
	CByteImage viz;
	TypeConvert(f, viz);
	return viz;
}

// ============================================================================
// HOG
// ============================================================================

static float derivKvals[3] = { -1, 0, 1};

HOGFeatureExtractor::HOGFeatureExtractor(int nAngularBins, bool unsignedGradients, int cellSize):
_nAngularBins(nAngularBins),
_unsignedGradients(unsignedGradients),
_cellSize(cellSize)
{
    _kernelDx.ReAllocate(CShape(3, 1, 1), derivKvals, false, 1);
    _kernelDx.origin[0] = 1;

    _kernelDy.ReAllocate(CShape(1, 3, 1), derivKvals, false, 1);
    _kernelDy.origin[0] = 1;

    // For visualization
    // A set of patches representing the bin orientations. When drawing a hog cell 
    // we multiply each patch by the hog bin value and add all contributions up to 
    // form the visual representation of one cell. Full HOG is achieved by stacking 
    // the viz for individual cells horizontally and vertically.
    _oriMarkers.resize(_nAngularBins);
    const int ms = 11;
    CShape markerShape(ms, ms, 1);

    // First patch is a horizontal line
    _oriMarkers[0].ReAllocate(markerShape, true);
    _oriMarkers[0].ClearPixels();
    for(int i = 1; i < ms - 1; i++) _oriMarkers[0].Pixel(/*floor(*/ ms/2 /*)*/, i, 0) = 1;

#if 0 // debug
	std::cout << "DEBUG:" << __FILE__ << ":" << __LINE__ << std::endl;
	for(int i = 0; i < ms; i++) {
		for(int j = 0; j < ms; j++) {
			std::cout << _oriMarkers[0].Pixel(j, i, 0) << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	char debugFName[2000];
	sprintf(debugFName, "/tmp/debug%03d.tga", 0);
	PRINT_EXPR(debugFName);
	WriteFile(_oriMarkers[0], debugFName);
#endif

	// The other patches are obtained by rotating the first one
	CTransform3x3 T = CTransform3x3::Translation((ms - 1) / 2.0, (ms - 1) / 2.0);
    for(int angBin = 1; angBin < _nAngularBins; angBin++) {
    	double theta;
    	if(unsignedGradients) theta = 180.0 * (double(angBin) / _nAngularBins);
    	else theta = 360.0 * (double(angBin) / _nAngularBins);
   		CTransform3x3 R  = T * CTransform3x3::Rotation(theta) * T.Inverse();

   		_oriMarkers[angBin].ReAllocate(markerShape, true);
   		_oriMarkers[angBin].ClearPixels();

		WarpGlobal(_oriMarkers[0], _oriMarkers[angBin], R, eWarpInterpLinear);

#if 0 // debug
		char debugFName[2000];
		sprintf(debugFName, "/tmp/debug%03d.tga", angBin);
		PRINT_EXPR(debugFName);
		WriteFile(_oriMarkers[angBin], debugFName);
#endif
    }
}


/*
	- Standing questions:
		- Do you take the max gradient, or do you perform operations for each channel?
		- How do you use orientation to weigh a pixel's contribution?
		- Normalization?
*/
Feature 
HOGFeatureExtractor::operator()(const CByteImage& img_) const
{
	/******** BEGIN TODO ********/
	// Compute the Histogram of Oriented Gradients feature
	// Steps are:
	// 1) Compute gradients in x and y directions
	CFloatImage convertedImg;
	TypeConvert(img_, convertedImg);

	CFloatImage derivX;
	CFloatImage derivY;

	CFloatImage magnitudeImg(convertedImg.Shape());
	CFloatImage orientationImg(convertedImg.Shape());

	Convolve(convertedImg, derivX, ConvolveKernel_SobelX);
	Convolve(convertedImg, derivY, ConvolveKernel_SobelY);

	// 2) Compute gradient magnitude and orientation
	for (int row = 0; row < convertedImg.Shape().height; row++)
	{
		for (int column = 0; column < convertedImg.Shape().width; column++)
		{
			// get max X and Y gradients
			float maxX = MAX(derivX.Pixel(column, row, 0), MAX(derivX.Pixel(column, row, 1), derivX.Pixel(column, row, 2)));
			float maxY = MAX(derivY.Pixel(column, row, 0), MAX(derivY.Pixel(column, row, 1), derivY.Pixel(column, row, 2)));

			// magnitude for x
			magnitudeImg.Pixel(column, row, 0) = sqrt(pow(maxX, 2) + pow(maxY, 2));
			// magnitude for y
			if (maxX == 0)
			{
				orientationImg.Pixel(column, row, 0) = (maxY >= 0)? PI/2. : -PI/2.;
			}
			else
			{
				orientationImg.Pixel(column, row, 0) = (maxX > 0)? atan(maxY / maxX) : atan(maxY / maxX) + 3.14;
			}
		}
	}
	// 3) Add contribution each pixel to HOG cells whose
	//    support overlaps with pixel
	// 4) Normalize HOG for each cell
	// 
	// Useful functions:
	// convertRGB2GrayImage, TypeConvert, WarpGlobal, Convolve, 

	printf("TODO: Feature.cpp:189\n"); exit(EXIT_FAILURE); 

	/******** END TODO ********/

}

CByteImage 
HOGFeatureExtractor::render(const Feature& f) const
{
	CShape cellShape = _oriMarkers[0].Shape();
	CFloatImage hogImgF(CShape(cellShape.width * f.Shape().width, cellShape.height * f.Shape().height, 1));
	hogImgF.ClearPixels();

	float minBinValue, maxBinValue;
	f.getRangeOfValues(minBinValue, maxBinValue);

	// For every cell in the HOG
	for(int hi = 0; hi < f.Shape().height; hi++) {
		for(int hj = 0; hj < f.Shape().width; hj++) {

			// Now _oriMarkers, multiplying contribution by bin level
			for(int hc = 0; hc < _nAngularBins; hc++) {
				float v = f.Pixel(hj, hi, hc) / maxBinValue;
				for(int ci = 0; ci < cellShape.height; ci++) {
					float* cellIt = (float*) _oriMarkers[hc].PixelAddress(0, ci, 0);
					float* hogIt = (float*) hogImgF.PixelAddress(hj * cellShape.height, hi * cellShape.height + ci, 0);

					for(int cj = 0; cj < cellShape.width; cj++, hogIt++, cellIt++) {
						(*hogIt) += v * (*cellIt);
					}
				}
			}
		
		}
	}

	CByteImage hogImg;
	TypeConvert(hogImgF, hogImg);
	return hogImg;
}


