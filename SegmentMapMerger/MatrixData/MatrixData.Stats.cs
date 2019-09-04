using System;
using System.Linq;
using System.Collections.Generic;

namespace MatrixOps
{
    public partial class MatrixData
    {
        //Finds the minimum value for the given column
        public double Min(int col)
        {
            return GetColumnCopy<double>(col).Min();
        }

        
        //Finds the maximum value for the given column
        public double Max(int col)
        {
            return GetColumnCopy<double>(col).Max();
        }

        //Find the Mean (average) value for the given column
        public double Mean(int col)
        {
            return GetColumnCopy<double>(col).Average();
        }
        public double MeanIf(int col,Func<double, bool> condition)
        {
            double[] vals = GetColumnCopy<double>(col);
            List<double> outList = new List<double>();
            foreach (double d in vals)
            {
                if (condition(d)) outList.Add(d);
            }   
            if(outList.Count > 0) return outList.Average();
            return 0;
        }

        //find the Mode of a column
        public double Mode(int col)
        {
            return GetColumnCopy<double>(col)
                .GroupBy(v => v)
                .OrderByDescending(g => g.Count())
                .First()
                .Key;
        }

        //Find the Median of a column 
        public double Median(int col)
        {
            double[] column = GetColumnCopy<double>(col)
                .OrderByDescending(g => g).ToArray();
            int mid = NumberOfRows / 2;
            return (NumberOfRows % 2 != 0) ? (double)column[mid] : ((double)column[mid] + (double)column[mid - 1]) / 2;
        }

        //Find the sum of a column
        public double Sum(int col)
        {
            return GetColumnCopy<double>(col).Sum();
        }

        public int CountIf(int col, Func<dynamic, dynamic, bool> condition, dynamic value)
        {
            dynamic[] colData = Columns(col);
            int counter = 0;
            foreach (dynamic d in colData)
            {
                if(condition(value, d)) counter++;
            }
            return counter;
        }
        

        public Dictionary<string, int> UniqueValues()
        {
            List<Dictionary<string, int>> uniqueVals = new List<Dictionary<string, int>>();
        
            for(int i = 0; i<NumberOfColumns;i++)
            {
                uniqueVals.Add(UniqueValues(i));
            }

            Dictionary<string, int> outList = new Dictionary<string, int>();
            foreach (Dictionary<string, int> d in uniqueVals)
            {
                foreach (KeyValuePair<string, int> kvp in d)
                {
                    if(!outList.ContainsKey(kvp.Key))
                    {
                        outList.Add(kvp.Key, 0);
                    }
                    outList[kvp.Key] += kvp.Value;
                }
            }
            return outList;
        }

        public Dictionary<string, int> UniqueValues(int col)
        {
            Dictionary<string, int> output = new Dictionary<string, int>();
            dynamic[] colval = Columns(col);
            dynamic uniqueVals = colval.Distinct().OrderBy(i => i);
            foreach (var v in uniqueVals)
            {                
                output.Add(Convert.ToString(v),colval.Count(c => object.Equals(c,v)));
            }
            return output;
        }    
    }
}