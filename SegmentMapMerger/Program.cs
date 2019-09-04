using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using MatrixOps;

namespace SegmentMapMerger
{
    class Program
    {
        static void Main(string[] args)
        {
            foreach (string s in args)
            {
                System.Console.WriteLine(s);
            }
            Console.WriteLine("Hello World!");
        }

        # region todo, make the methods below work with this app
        static void processDir(string dir)
        {
             MatrixData classes = new MatrixData("classes.csv",true,true,',');

            List<string> files = DirSearch(dir);
            string baseFile = GetBaseImage(files);
            files.Remove(baseFile);

            System.Console.WriteLine("Reading file: " + baseFile);
            List<MatrixData> baseImage = MatrixDataImageExtensions.LoadImageAsMatrix(baseFile);
            Dictionary<int, MatrixData> masks = new Dictionary<int, MatrixData>();
            foreach (string maskImagePath in files)
            {
                System.Console.WriteLine("\tProcessing file: " + maskImagePath);
                string cls = GetClass(maskImagePath);
                int valToWrite = int.Parse(classes.GetValueOfCell("_id","_class",cls.ToLower()).ToString());
                MatrixData mask = MatrixDataImageExtensions.LoadImageAsMatrix(maskImagePath).First().ToBinaryByVal(255,true);
                masks.Add(valToWrite,mask);
                System.Console.WriteLine("\t\tFound class \""+ cls+"\", mask created with value of: " + valToWrite);
            }
            MatrixData outputY = new MatrixData(baseImage[0].NumberOfRows,baseImage[0].NumberOfColumns); outputY.SetAll(() => {return 0;});
            System.Console.Write("\tCreating output...");
            foreach(int key in masks.Keys.ToArray())
            {
                outputY = outputY.WriteValOnMask(masks[key],key);
            }

            System.Console.WriteLine("Done");
            string originalFileName = baseFile.Split('\\').Last();
            string outFileName = originalFileName.Split('.').First()+".png";
            string outPath = baseFile.Replace(originalFileName,"");
            outputY.ToImage(outPath+outFileName);
            System.Console.WriteLine("\tCreated: " + outPath+outFileName);
        }

        private static string GetClass(string input)
        {
            return input.Split('\\').Last().Split('.').First().Split('_').Last();
        }
        private static string GetBaseImage(List<string> input)
        {
            return input.Where(i => i.EndsWith(".jpg")).First();
        }
        private static List<string> DirSearch(string sDir)
        {
            List<string> files = new List<string>();
            foreach (string f in Directory.GetFiles(sDir))
            {
                files.Add(f);
            }
            foreach (string d in Directory.GetDirectories(sDir))
            {
                files.AddRange(DirSearch(d));
            }
            return files;
        }
        # endregion
    }
}
