#ifndef FEATURE_H
#define FEATURE_H

#include "Common.h"
#include "Utils.h"
#include "ImageDatabase.h"

typedef CFloatImage Feature;
typedef std::vector<Feature> FeatureSet;

// Abstract super class for all feature extractors. 
class FeatureExtractor
{
public:
	FeatureExtractor() {};

	// Extract feature vector for image image. Decending classes must implement this method
	virtual Feature operator()(const CByteImage& image) const = 0;

	// Extracts descripto for all images in dataset, stores result in featureSet
	void operator()(const ImageDatabase& db, FeatureSet& featureSet) const;

	// Generate a visualization for the feature f, for debuging and inspection purposes only.
	virtual CByteImage render(const Feature& f) const = 0;

	// Same as render(f) but normalizes values to be in range (0,1) by dividing by max value
	CByteImage render(const Feature& f, bool normalizeFeat) const;
};

// Factory method that allocates the correct feature vector extractor given
// the name of the extractor (caller is responsible for deallocating 
// extractor). To extend the code with other feature extractors or 
// other configurations for feature extractors add appropriate constructor
// calls in implementation of this function.
FeatureExtractor* FeatureExtractorNew(const char* featureType);

// Tiny Image feature. Converts image to grayscale and downscales it. Used
// mostly as a baseline feature.
class TinyImageFeatureExtractor : public FeatureExtractor
{
private:
	int _targetW, _targetH;

public:	
	TinyImageFeatureExtractor(int targetWidth = 16, int targetHeight = 32);
	Feature operator()(const CByteImage& image) const;

	CByteImage render(const Feature& f) const;
};

// Histogram of Oriented Gradients feature.
class HOGFeatureExtractor : public FeatureExtractor
{
private:
	int _nAngularBins;                    // How many angular bins are there
	bool _unsignedGradients;              // If true then we only consider the orientation modulo 180 degrees (i.e., 190 
		                                  // degrees is considered the same as 10 degrees)
	int _cellSize;                        // Support size of a cell, in pixels
    CFloatImage _kernelDx, _kernelDy;     // Derivative kernels in x and y directions
    std::vector<CFloatImage> _oriMarkers; // Used for visualization

public:
	HOGFeatureExtractor(int nAngularBins = 18, bool unsignedGradients = true, int cellSize = 6);

	Feature operator()(const CByteImage& image) const;

	CByteImage render(const Feature& f) const;
};

#endif
