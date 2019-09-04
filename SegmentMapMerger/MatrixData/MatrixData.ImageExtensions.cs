using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace MatrixOps
{
    public static class MatrixDataImageExtensions
    {
        private static int[,] sobelX = new int[,]{
                {-1,0,1},
                {-1,0,1},
                {-1,0,1}
            };
        private static int[,] sobelY = new int[,]{
                { 1, 1, 1},
                { 0, 0, 0},
                {-1,-1,-1}
            };
        private static int[,] embossfilter = new int[,]{
                {0,0,0},
                {1,0,-1},
                {0,0,0}
            };
        private static int[,] embossfilter1 = new int[,]{
                {1,0,0},
                {0,0,0},
                {0,0,-1}
            };
        private static int[,] embossfilter2 = new int[,]{
                {0,1,0},
                {0,0,0},
                {0,-1,0}
            };
        private static int[,] embossfilter3 = new int[,]{
                {0,0,1},
                {0,0,0},
                {-1,0,0}
            };

        public static void ToImage(this MatrixData input, string fileName)
        {
            using (Image<Rgba32> image = new Image<Rgba32>(input.NumberOfRows, input.NumberOfColumns))
            {
                for (int r = 0; r < input.NumberOfRows; r++)
                {
                    for (int c = 0; c < input.NumberOfColumns; c++)
                    {
                        byte b = byte.Parse(((double)input[r, c]).ToString());
                        image[r, c] = ColorBuilder<Rgba32>.FromRGB(b, b, b);
                    }
                }
                image.SaveAsBmp(File.OpenWrite(fileName));
            }
        }

        public static void ToRGBImage(this List<MatrixData> input, string fileName)
        {
            MatrixData R = input[0];
            MatrixData G = input[1];
            MatrixData B = input[2];
            using (Image<Rgba32> image = new Image<Rgba32>(R.NumberOfRows, R.NumberOfColumns))
            {
                for (int r = 0; r < R.NumberOfRows; r++)
                {
                    for (int c = 0; c < R.NumberOfColumns; c++)
                    {
                        byte red = byte.Parse(((double)R[r, c]).ToString());
                        byte green = byte.Parse(((double)G[r, c]).ToString());
                        byte blue = byte.Parse(((double)B[r, c]).ToString());
                        image[r, c] = ColorBuilder<Rgba32>.FromRGB(red, green, blue);
                    }
                }
                image.SaveAsBmp(File.OpenWrite(fileName));
            }
        }
        public static List<MatrixData> LoadImageAsMatrix(string fileName) 
        {
            Image<Rgba32> image = Image.Load(fileName);
            MatrixData R = new MatrixData(image.Width,image.Height);
            MatrixData G = new MatrixData(image.Width,image.Height);
            MatrixData B = new MatrixData(image.Width,image.Height);
            for (int r = 0; r < image.Width; r++)
            {
                for (int c = 0; c < image.Height; c++)
                {                    
                    R[r,c] = double.Parse(image[r,c].R.ToString());
                    G[r,c] = double.Parse(image[r,c].G.ToString());
                    B[r,c] = double.Parse(image[r,c].B.ToString());
                }
            }
            image.Dispose();
            return new List<MatrixData>(){R,G,B};
        }
        public static MatrixData Test (List<MatrixData> input, int rad, double bP = .5)
        {
            MatrixData R = input[0].HistEQ();            
            MatrixData G = input[1].HistEQ();
            MatrixData B = input[2].HistEQ();

            Dictionary<string, int> rHist = R.GetHist();
            Dictionary<string, int> gHist = G.GetHist();
            Dictionary<string, int> bHist = B.GetHist();
            Dictionary<string, double> hist = new Dictionary<string, double>();
            foreach (string key in rHist.Keys)
            {
                hist.Add(key, rHist[key]+gHist[key]+bHist[key]);
            }
            double max = hist.Values.Max();
            double min = hist.Values.Min();
            string[] keys = hist.Keys.ToArray();
            foreach (string key in keys)
            {
                hist[key] = PreformStandardScoreNormalization(hist[key],min,max);
            }

            MatrixData Output = new MatrixData(R.NumberOfRows,R.NumberOfColumns);
            for (int r = 0; r < input[1].NumberOfRows; r++)
            {
                for (int c = 0; c < input[1].NumberOfColumns; c++)
                {
                    R[r,c] = R[r,c] * (1 - (hist[int.Parse(Convert.ToString(R[r,c])).ToString()])*bP);
                    G[r,c] = G[r,c] * (1 - (hist[int.Parse(Convert.ToString(G[r,c])).ToString()])*bP);
                    B[r,c] = B[r,c] * (1 - (hist[int.Parse(Convert.ToString(B[r,c])).ToString()])*bP);
                }
            }
            for (int r = rad; r < input[1].NumberOfRows-rad; r++)
            {
                for (int c = rad; c < input[1].NumberOfColumns-rad; c++)
                {
                    double dev = Math.Pow(((rad*2)+1),2);
                    double rt = SumRad(R,r,c,rad)/dev;
                    double gt = SumRad(G,r,c,rad)/dev;
                    double bt = SumRad(B,r,c,rad)/dev;
                    /* =(R[r-1,c-1] + R[r-1,c] + R[r-1,c+1] + 
                                 R[r,c-1] +              R[r,c+1] +
                                R[r+1,c-1] + R[r+1,c] + R[r+1,c+1])/8;

                    double gt =(G[r-1,c-1] + G[r-1,c] + G[r-1,c+1] + 
                                G[r,c-1] +              G[r,c+1] +
                                G[r+1,c-1] + G[r+1,c] + G[r+1,c+1])/8;

                    double bt =(B[r-1,c-1] + B[r-1,c] + B[r-1,c+1] + 
                                B[r,c-1] +              B[r,c+1] +
                                B[r+1,c-1] + B[r+1,c] + B[r+1,c+1])/8; */
                    
                    //double mm = Bin(R[r,c],G[r,c],B[r,c],5);
                    double mm = Bin(rt,gt,bt,5);                    
                    Output[r,c] = mm;
                    /*
                    Output[r,c] = (Math.Abs(((R[r,c] + G[r,c] + B[r,c])/3))) +
                    
                    ((Math.Abs(((R[r,c+1] + G[r,c+1] + B[r,c+1])/3)) +
                    Math.Abs(((R[r+1,c] + G[r+1,c] + B[r+1,c])/3)) +
                    Math.Abs(((R[r,c-1] + G[r,c-1] + B[r,c-1])/3)) +
                    Math.Abs(((R[r-1,c] + G[r-1,c] + B[r-1,c])/3)))/4);
                    Output[r,c] = (Math.Abs(((R[r,c] + G[r,c] + B[r,c])/3)) - Math.Abs(((R[r,c+1] + G[r,c+1] + B[r,c+1])/3)) +
                    Math.Abs(((R[r,c] + G[r,c] + B[r,c])/3)) - Math.Abs(((R[r+1,c] + G[r+1,c] + B[r+1,c])/3)) +
                    Math.Abs(((R[r,c] + G[r,c] + B[r,c])/3)) - Math.Abs(((R[r,c-1] + G[r,c-1] + B[r,c-1])/3)) +
                    Math.Abs(((R[r,c] + G[r,c] + B[r,c])/3)) - Math.Abs(((R[r-1,c] + G[r-1,c] + B[r-1,c])/3)));*/
                }
            }
            return Output;
        }

        private static double PreformStandardScoreNormalization(double val, double min, double max)
        {
            double mult = (max == min)? 1 : 1 / (max - min);

            double currentVal = val;
            currentVal = (currentVal - min) * mult;
            return currentVal;
        }
        public static double Bin(double r, double g, double b, int bins, double min = 0, double max = 255) 
        {
            double width = (max - min) / double.Parse(bins.ToString());
            double[] intrivals = new double[bins+1];
            
            for (int i = 0; i<bins+1;i++)
            {
                intrivals[i] = width*i;
            }



            double val =(r+b+g)/3;
            double bri = intrivals.ClosestTo(val);
            //return (bri+red+green+blue);
            return bri;
            throw new Exception();
        }
        public static Dictionary<string, int> GetHist (this MatrixData input)
        {
            return input.UniqueValues();
        }
        public static MatrixData FillBinaryByRadius(this MatrixData input,int rad,double t = 0.5)
        {
            
            MatrixData output = new MatrixData(input.NumberOfRows,input.NumberOfColumns);
            for (int row = rad; row < input.Data.GetLength(0)-rad;row++)
            {
                for (int col = rad; col< input.Data.GetLength(1)-rad;col++)
                {
                    
                    output[row,col] = (SumRad(input,row,col,rad) / Math.Pow(((rad*2)+1),2) > t)? 1:0;
                }
            }
            return output;
        }
        public static double SumRad(MatrixData input, int row, int col, int rad)
        {
            double output = 0;
            for (int r = row-rad; r < row+rad;r++)
            {
                for (int c = col-rad; c< col+rad;c++)
                {
                    output += input[r,c];
                }
            }
            return output;
        }
        public static double ClosestTo(this IEnumerable<double> collection, double target)
        {
            // NB Method will return int.MaxValue for a sequence containing no elements.
            // Apply any defensive coding here as necessary.
            double closest = double.MaxValue;
            var minDifference = int.MaxValue;
            foreach (var element in collection)
            {
                var difference = Math.Abs((long)element - target);
                if (minDifference > difference)
                {
                    minDifference = (int)difference;
                    closest = element;
                }
            }

            return closest;
        }
        
        public static double MaxOf(params double[] input) {
            double max = input.First();
            foreach(double d in input) {
                if (d > max) max = d;
            }
            return max;
        }
        public static MatrixData Exp(this MatrixData input)
        {
            MatrixData output = new MatrixData(input.NumberOfRows,input.NumberOfColumns);
            for (int row = 0; row < input.Data.GetLength(0);row++)
            {
                for (int col = 0; col< input.Data.GetLength(1);col++)
                {
                    output[row,col] = Math.Pow(input[row,col],2);
                }
            }
            return output;
        }

        public static MatrixData RoundAll(this MatrixData input)
        {
            for (int c = 0; c<input.NumberOfColumns;c++)
            {
                for (int r = 0; r < input.NumberOfRows; r++)
                {
                    input[r, c] = Math.Round((double)input[r, c], 0);
                }
            }
            return input;
        }
        public static MatrixData Fill(this MatrixData input)
        {
            for (int c = 0; c<input.NumberOfColumns;c++)
            {
                for (int r = 0; r < input.NumberOfRows; r++)
                {
                    try
                    {
                        byte b = byte.Parse(((double)input[r, c]).ToString());
                    }
                    catch {
                        input[r, c] = 0;
                    }
                }
            }
            return input;
        }
        public static MatrixData ToBinaryByVal(this MatrixData input, double val,bool invert = false)
        {
            MatrixData output = new MatrixData(input.NumberOfRows,input.NumberOfColumns);
            int trueval = (!invert)?  1 : 0;
            int falseval = (!invert)?  0 : 1;
            for (int c = 0; c<input.NumberOfColumns;c++)
            {
                for (int r = 0; r < input.NumberOfRows; r++)
                {
                    output[r, c] =  (Between(input[r, c], val-1,val+1))? trueval:falseval;
                    
                }
            }
            return output;
        }
        public static MatrixData WriteValOnMask(this MatrixData input, MatrixData mask,int value)
        {
            MatrixData output = new MatrixData(input.NumberOfRows,input.NumberOfColumns);

            for (int c = 0; c<input.NumberOfColumns;c++)
            {
                for (int r = 0; r < input.NumberOfRows; r++)
                {
                    if(mask[r, c] == 0)
                    {
                        output[r, c] = input[r, c];
                    } else {
                        output[r, c] = (double)value;
                    }                   
                }
            }
            return output;
        }
        public static bool Between(double num, double lower, double upper, bool inclusive = false)
        {
            return inclusive
                ? lower <= num && num <= upper
                : lower < num && num < upper;
        }
        public static MatrixData Clamp(this MatrixData input, int min, int max)
        {
            for (int c = 0; c<input.NumberOfColumns;c++)
            {
                for (int r = 0; r < input.NumberOfRows; r++)
                {
                    input[r, c] = ((double)input[r, c]).Clamp(min,max);
                }
            }
            return input;
        }
        public static double Clamp(this double input, int min, int max)
        {
            if(input < min)
            {
                return min;
            } 
            else if(input > max) 
            {
                return max;
            }
            else return Math.Round(input, 0);
        }
        public static MatrixData GetSobelEdge(this MatrixData input)
        {
            MatrixData output = new MatrixData(input.NumberOfRows,input.NumberOfColumns);          
            MatrixData sobelXimg = input.ApplyConvolutionFilter(sobelX);
            MatrixData sobelYimg = input.ApplyConvolutionFilter(sobelY);
            for(int r = 0; r<input.NumberOfRows;r++)
            {
                for (int c = 0; c<input.NumberOfColumns; c++)
                {
                    output[r,c] = Math.Sqrt(Math.Pow(sobelXimg[r,c],2) + Math.Pow(sobelYimg[r,c],2));
                    if(output[r,c] > 255) output[r,c] = 255;
                }
            }
            return output;            
        }
        public static MatrixData GetEmbossEdge(this MatrixData input)
        {            
            MatrixData d = input.ApplyConvolutionFilter(embossfilter);
            MatrixData d1 = input.ApplyConvolutionFilter(embossfilter1);
            MatrixData d2 = input.ApplyConvolutionFilter(embossfilter2);
            MatrixData d3 = input.ApplyConvolutionFilter(embossfilter3);
            MatrixData output = d.SubtractIfLessThanMode(d1).SubtractIfLessThanMode(d2).SubtractIfLessThanMode(d3);
            return output;
        }
        public static MatrixData HistEQ(this MatrixData input)
        {
            MatrixData output = new MatrixData(input.Data);
            output.NormalizeAll(NormalizationMethod.StandardScore,true);
            output = output*255;
            output = output.Clamp(0,255);
            return output;
        }
        public static int Width (this MatrixData m) => m.NumberOfColumns;
        public static int Height (this MatrixData m) => m.NumberOfRows;
        public static MatrixData ReduceDimensionByOne(this MatrixData input, int dimension)
        {
            bool reduceRow= dimension==0;
            int height =(reduceRow)? input.Height()-1 : input.Height();
            int width =(reduceRow)?  input.Width() : input.Width()-1;
            MatrixData output = new MatrixData(height,width);
            Random rand = new Random();

            int[] rToSkip = Enumerable
                .Repeat(0, width)
                .Select(i => rand.Next(0, width))
                .ToArray();

            int[] cToSkip = Enumerable
                .Repeat(0, height)
                .Select(i => rand.Next(0, height))
                .ToArray();

            
            int rInIndex = 0;
            int cInIndex = 0;

            if (reduceRow)
            {
                int rowToSkip;
                for (int c = 0; c<width; c++)
                {
                    rowToSkip = rToSkip[c];
                    rInIndex = 0;
                    for (int r = 0; r<height;r++)
                    {
                        if (r == rowToSkip) rInIndex++;
                        output[r,c] = input[rInIndex, cInIndex];
                        rInIndex++;
                    }
                    cInIndex++;
                }
            } 
            else
            {
                int colToSkip;
                for (int r = 0; r<height;r++)
                {
                    colToSkip = cToSkip[r];
                    cInIndex = 0;
                    for (int c = 0; c<width; c++)
                    {
                        if (c == colToSkip) cInIndex++;
                        output[r,c] = input[rInIndex, cInIndex];
                        cInIndex++;
                    }
                    rInIndex++;
                }
            }            
            return output;
        }
        public static MatrixData Convert18To6Px(this MatrixData input, bool favorBrightPx = true)
        {
            MatrixData output = new MatrixData(6,6);
            int outR = 0;
            int outC = 0;
            for (int r = 0; r<input.Height();r+=3)
            {
                outC = 0;
                for (int c = 0; c<input.Width(); c+=3)
                {
                    MatrixData m = input.CopyData(r,c,3,3);   
                    output[outR,outC] = (favorBrightPx)? (m.MeanPixel()+m.Percentile(25))/2 : m.MeanPixel();
                    outC++;
                }
                outR++;
            }
            return output;
        }
        public static MatrixData CropEdges(this MatrixData input, int pixels)
        {
            MatrixData output = new MatrixData(input.NumberOfRows-(pixels*2), input.NumberOfColumns-(pixels*2));
            for (int r = pixels; r<input.NumberOfRows-pixels;r++)
            {
                for (int c = pixels; c<input.NumberOfRows-pixels;c++)
                {
                    output[r-pixels, c-pixels] = input[r,c];
                }
            }
            return output;

        }
        //http://www.gutgames.com/post/Matrix-Convolution-Filters-in-C.aspx
        public static MatrixData ApplyConvolutionFilter(this MatrixData Input, int[,] filter)
        {
            MatrixData NewBitmap = new MatrixData(Input.Width(), Input.Height());
            MatrixData OldData = Input;
            double MeanPixel = Input.ModePixel();
            int Width = filter.GetLength(1);
            int Height = filter.GetLength(0);
            for (int x = 0; x < Input.Width(); ++x)
            {
                for (int y = 0; y < Input.Height(); ++y)
                {
                    double Value = 0;
                    double Weight = 0;
                    int XCurrent = -Width / 2;
                    for (int x2 = 0; x2 < Width; ++x2)
                    {
                        if (XCurrent + x < Input.Width() && XCurrent + x >= 0)
                        {
                            int YCurrent = -Height / 2;
                            for (int y2 = 0; y2 < Height; ++y2)
                            {
                                if (YCurrent + y < Input.Height() && YCurrent + y >= 0)
                                {
                                    double Pixel;
                                    
                                    try
                                    {
                                        Pixel = Input[YCurrent + y, YCurrent + x];
                                    }
                                    catch
                                    {

                                        Pixel = MeanPixel;
                                    } 
                                    Value += filter[x2, y2] * Pixel;
                                    Weight += filter[x2, y2];
                                }
                                ++YCurrent;
                            }
                        }
                        ++XCurrent;
                    }
                    double PixelVal = Input[y,x];
                    if (Weight == 0)
                        Weight = 1;
                    if (Weight > 0)
                    {
                        Value = System.Math.Abs(Value);                        
                        Value = (Value/Weight);
                        Value = (Value>255)? 255 : Value;
                        PixelVal = Value;
                    }
                    NewBitmap[y,x] = PixelVal;
                }
            }
            return NewBitmap;
        }
        public static MatrixData UpScale(this MatrixData input, int factor)
        {
            MatrixData output = new MatrixData(input.Height()*factor, input.Width()*factor);

            for (int r = 0;r<input.NumberOfRows;r++)
            {
                for (int c = 0;c<input.NumberOfColumns;c++)
                {               
                    for (int oR = r*factor;oR<(r*factor)+factor;oR++)
                    {
                        for (int oC = c*factor;oC<(c*factor)+factor ;oC++)
                        {
                            output[oR,oC] = input[r,c];
                        }
                    }
                }
            }
            return output;
        }
        public static MatrixData DownScale(this MatrixData input, int factor,bool favorBrightPx = true)
        {
            MatrixData output = new MatrixData(input.Height()/factor, input.Width()/factor);
            for (int r = 0;r<output.NumberOfRows;r++)
            {
                for (int c = 0;c<output.NumberOfColumns;c++)
                {
                    MatrixData inpx = new MatrixData(input,r*factor, c*factor, factor,factor);
                    output[r,c] = (favorBrightPx)? (inpx.MeanPixel() + inpx.Percentile(25))/2: inpx.MeanPixel();
                }
            }
            return output;
        }
        public static double MeanPixel (this MatrixData input)
        {
            double[] colMean = new double[input.NumberOfColumns];
            for (int c = 0; c<input.NumberOfColumns; c++)
            {
                colMean[c] = input.Mean(c);
            }
            return colMean.Average();
        }
        public static double MeanIfPixel (this MatrixData input,Func<double, bool> condition)
        {
            List<double> colMean = new List<double>();
            for (int c = 0; c<input.NumberOfColumns; c++)
            {
                double d = input.MeanIf(c,condition);
                if(condition(d))
                {
                    colMean.Add(d);
                }
            }
            return colMean.Average();
        }

        public static double ModePixel (this MatrixData input)
        {
            double[] colMode = new double[input.NumberOfColumns];
            for (int c = 0; c<input.NumberOfColumns; c++)
            {
                colMode[c] = input.Mode(c);
            }
            return colMode.Average();
        }
        public static MatrixData SubtractIfLessThanMode(this MatrixData input, MatrixData input1)
        {
            MatrixData output = new MatrixData(input.NumberOfRows,input.NumberOfColumns);
            double[] colMode = new double[input.NumberOfColumns];
            for (int c = 0; c<input.NumberOfColumns; c++)
            {
                colMode[c] = input1.Mode(c);
            }
            double modeVal = colMode.GroupBy(v => v)
                .OrderByDescending(g => g.Count())
                .First()
                .Key;
            
            for (int r = 0;r<input.NumberOfRows;r++)
            {
                for (int c = 0;c<input.NumberOfColumns;c++)
                {        
                    output[r,c] = (input1[r,c] < modeVal)? input[r,c] - input1[r,c]: input[r,c];
                }
            }
            return output;
        }

        public static MatrixData NormalizeAllBetween(this MatrixData input, double min, double max)
        {
            MatrixData output = new MatrixData(input,0,0);
            double mult = (max == min)? 1 : 1 / (max - min);
            for (int col = 0; col < input.NumberOfColumns; col++)
            {
                for (int row = 0; row < input.NumberOfRows; row++)
                {
                    double currentVal = input[row, col];
                    currentVal = (currentVal - min) * mult;
                    output[row, col] = currentVal;
                }
            }
            return output;
        }
        public static MatrixData ToBinary(this MatrixData input, double divider)
        {
            MatrixData output = new MatrixData(input,0,0);
            for(int r = 0; r < output.NumberOfRows; r++)
            {
                for (int c = 0; c<output.NumberOfColumns; c++)
                {
                    output[r,c] = (output[r,c]>divider)? 255: 0;
                    
                }
            }

            return output;
        }

        public static MatrixData CalculateDistanceField(this MatrixData input, int searchDistance)
        {
            MatrixData output = new MatrixData(input.NumberOfRows,input.NumberOfColumns);
            for (int r = 0; r < input.NumberOfRows; r++)
            {
                for(int c = 0; c < input.NumberOfColumns; c++) 
                {
                    //skip black pixels
                    if (input[r,c]==0) continue;

                    int steps = 0;
                    int fxMin = Math.Max(r - searchDistance, 0);
                    int fxMax = Math.Min(r + searchDistance, input.NumberOfRows);

                    int fyMin = Math.Max(c - searchDistance, 0);
                    int fyMax = Math.Min(c + searchDistance, input.NumberOfColumns);
                    
                    for (int fx = fxMin; fx < fxMax-1; ++fx)
                    {
                        for (int fy = fyMin; fy < fyMax-1; ++fy)
                        {
                            if(input[fx,fy]==255)
                            {
                                steps +=((fx - fxMin) + (fy - fyMin));
                                //int tempStep = ((fx - fxMin) + (fy - fyMin));
                                //if(tempStep > steps) steps = tempStep;
                            }
                        }
                    }
                    output[r,c] = steps; 
                }
            }
            return output.HistEQ();
        }

        public static MatrixData GetLargestBlob(this MatrixData input, ref Tuple<int, int> blobMid)
        {
            MatrixData output = new MatrixData(input.NumberOfRows,input.NumberOfColumns);            
            List<Tuple<int,int>> checkedPixels = new List<Tuple<int, int>>();
            List<Tuple<int,int>> blobs = new List<Tuple<int, int>>();
            List<Tuple<int,int>> largestBlob = new List<Tuple<int, int>>();            
            Random rand = new Random();

            for (int r = 0; r < input.NumberOfRows; r++)
            {
                for(int c = 0; c < input.NumberOfColumns; c++) 
                {
                    //skip black pixels
                    if (input[r,c]==0) continue;

                    int replace = rand.Next(256,int.MaxValue);
                    

                    if (checkedPixels.Count==0)
                    {
                        List<Tuple<int,int>> blob = new List<Tuple<int, int>>();
                        blobs.Add(new Tuple<int, int>(Fill(input,r,c,replace,ref checkedPixels,ref blob), replace)); 
                        if(blob.Count > largestBlob.Count) largestBlob = blob;
                    } 
                    else
                    {
                        if(!checkedPixels.Contains(new Tuple<int, int>(r, c)))
                        {
                            List<Tuple<int,int>> blob = new List<Tuple<int, int>>();
                            blobs.Add(new Tuple<int, int>(Fill(input,r,c,replace,ref checkedPixels,ref blob), replace));
                            if(blob.Count > largestBlob.Count) largestBlob = blob;
                        }
                    }                    
                }
            }
            
            //set the output to only show the largest blob
            Tuple<int,int> largest = blobs.Where(r => r.Item1 == (blobs.Max(m => m.Item1))).First();
            for (int r = 0; r < input.NumberOfRows; r++)
            {
                for(int c = 0; c < input.NumberOfColumns; c++) 
                {
                    output[r,c] = (input[r,c]==largest.Item2)? 255 : 0;
                }
            }

            //calculate the center of the largest blob
            int xMin = largestBlob.Min(m => m.Item1);
            int xMax = largestBlob.Max(m => m.Item1);
            int yMin = largestBlob.Min(m => m.Item2);
            int yMax = largestBlob.Max(m => m.Item2);            
            int centerX = ((xMax - xMin) / 2) + xMin;
            int centerY = ((yMax - yMin) / 2) + yMin;
            blobMid = new Tuple<int, int>(centerX, centerY);

            
            return output;
        }

        public static int Fill(MatrixData array, int x, int y, int newInt, ref List<Tuple<int,int>> checkedPixels,ref List<Tuple<int,int>> blobPixels)
        {
            int initial = array[x,y];
            int counter = 0;
            Queue<Tuple<int,int>> queue = new Queue<Tuple<int,int>>();
            queue.Enqueue(new Tuple<int, int>(x, y));

            while (queue.Any())
            {
                Tuple<int, int> point = queue.Dequeue();

                if (array[point.Item1, point.Item2] != initial)
                    continue;

                array[point.Item1, point.Item2] = newInt;
                checkedPixels.Add(point);
                blobPixels.Add(point);
                counter++;
                EnqueueIfMatches(array, queue, point.Item1 - 1, point.Item2, initial);
                EnqueueIfMatches(array, queue, point.Item1 + 1, point.Item2, initial);
                EnqueueIfMatches(array, queue, point.Item1, point.Item2 - 1, initial);
                EnqueueIfMatches(array, queue, point.Item1, point.Item2 + 1, initial);        
            }
            return counter;
        }
        public static string AppendTimeStamp(this string fileName)
        {
            return string.Concat(
                Path.GetFileNameWithoutExtension(fileName),
                DateTime.Now.ToString("yyyyMMddHHmmssfff"),
                Path.GetExtension(fileName)
                );
        }
        private static void EnqueueIfMatches(MatrixData array, Queue<Tuple<int, int>> queue, int x, int y, int initial)
        {
            if (x < 0 || x >= array.Data.GetLength(0) || y < 0 || y >= array.Data.GetLength(1))
                return;

            if (array[x, y] == initial)
            queue.Enqueue(new Tuple<int, int>(x, y));
        }
    
    }    
}