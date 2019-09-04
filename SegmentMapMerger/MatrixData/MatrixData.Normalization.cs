using System;
using System.ComponentModel;
using System.Collections.Generic;

namespace MatrixOps
{
    public partial class MatrixData
    {
        //attempt to normalize a numaric column
        public void Normalize(int col, NormalizationMethod method = NormalizationMethod.StandardScore)
        {
            //https://docs.microsoft.com/en-us/azure/machine-learning/studio-module-reference/normalize-data
            //https://en.wikipedia.org/wiki/Normalization_(statistics)
            //https://en.wikipedia.org/wiki/Feature_scaling
            switch (method)
            {
                case NormalizationMethod.StandardScore:
                    PreformStandardScoreNormalization(col);
                    break;
                case NormalizationMethod.FeatureScalingStandardization:
                    PreformStandardization(col);
                    break;
                case NormalizationMethod.FeatureScalingMinMax:
                    PreformMinMaxNormalization(col);
                    break;
                case NormalizationMethod.FeatureScalingMean:
                    PreformMeanNormalization(col);
                    break;
                default:
                    throw new Exception("How did you even get here? Please let the monkey that" +
                    "coded this know the following: \"MatrixData<T>.Normalize.default\", along with" +
                    "what you did to cause this error");
            }
        }
        public void NormalizeAll(NormalizationMethod method = NormalizationMethod.StandardScore, bool ignoreDT = false)
        {
            for (int col = 0; col < NumberOfColumns; col++)
            {
                if (IsValueNumaric(col)|ignoreDT)
                {
                    Normalize(col, method);
                }
            }
        }

        private void PreformStandardScoreNormalization(int col)
        {
            //https://en.wikipedia.org/wiki/Standard_score
            double min = Min(col);
            double max = Max(col);

            double mult = (max == min)? 1 : 1 / (max - min);

            for (int row = 0; row < NumberOfRows; row++)
            {
                double currentVal = _data[row, col];
                currentVal = (currentVal - min) * mult;
                _data[row, col] = ConvertToNumeric(currentVal.ToString());
            }
        }

        private void PreformMinMaxNormalization(int col)
        {
            //https://en.wikipedia.org/wiki/Feature_scaling
            double min = Min(col);
            double max = Max(col);
            for (int row = 0; row < NumberOfRows; row++)
            {
                double currentVal = _data[row, col];
                currentVal = (currentVal - min) / (max - min);
                _data[row, col] = ConvertToNumeric(currentVal.ToString());
            }
        }

        private void PreformMeanNormalization(int col)
        {
            //https://en.wikipedia.org/wiki/Feature_scaling
            double min = Min(col);
            double max = Max(col);
            double mean = Mean(col);

            for (int row = 0; row < NumberOfRows; row++)
            {
                double currentVal = _data[row, col];
                currentVal = (currentVal - mean) / (max - min);
                _data[row, col] = ConvertToNumeric(currentVal.ToString());
            }
        }

        private void PreformStandardization(int col)
        {
            //https://en.wikipedia.org/wiki/Feature_scaling
            PreformStandardScoreNormalization(col);
        }

    }
}