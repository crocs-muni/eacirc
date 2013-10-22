import System.FilePath (pathSeparator)
import System.Posix.Files (fileExist)
import System.Environment (getArgs)
import Data.List (isInfixOf, intercalate)
import Control.Monad (foldM)

scoreFilename :: String
scoreFilename = "scores.log"

logFilename :: String
logFilename = "eacirc.log"

resultsFilename :: String
resultsFilename = "results.txt"

testVectorChangeFreq :: Int
testVectorChangeFreq = 100

stableGenTheshold :: Float
stableGenTheshold = 0.99

maxRuns :: Int
maxRuns = 30

logSuspects :: [String]
logSuspects = ["] warning: ","] error: "]

data GenerationRunInfo = Info {
  generationRun :: Int,
  avgFit :: Float,
  maxFit :: Float,
  minFit :: Float
} deriving (Show, Eq)

type Scores=[GenerationRunInfo]

data RunStats = Stats {
  errors :: [String],
  stableGen :: [Int],
  avgTable :: [GenerationRunInfo]
} deriving (Show, Eq)

emptyRunStats :: RunStats
emptyRunStats = Stats{errors=[], stableGen=[], avgTable=[]}

foldRunStats :: String -> RunStats -> Int -> IO RunStats
foldRunStats path stats run = do
  newStats <- processRun path run
  return Stats{errors=(concatMap errors [stats,newStats]), 
               stableGen=(concatMap stableGen [stats,newStats]),
               avgTable=(concatMap avgTable [stats,newStats])}

processRun :: String -> Int -> IO RunStats
processRun path run = do
  fileExist (path ++ show run) >>= \exists -> if exists 
    then do
      errors <- checkErrors path run
      fileExist scoreFile >>= \exists -> if exists
        then do
          scores <- readFile scoreFile >>= return . parseScoreFile
          return Stats{errors=errors, 
                       stableGen=[getStableGen scores], 
                       avgTable=[getAvgTable run scores]}
        else
          return Stats{errors=errors, stableGen=[], avgTable=[Info{generationRun=run,
                       avgFit= -1, maxFit= -1, minFit= -1}]}
    else return Stats{errors=["Run " ++ show run ++ " does not exist!"],
                      stableGen=[],
                      avgTable=[]}
  where scoreFile = path ++ show run ++ (pathSeparator : scoreFilename)

parseScoreFile :: String -> Scores
parseScoreFile = zipWith build [1..] . map (take 4 . words) . lines
  where build :: Int -> [String] -> GenerationRunInfo
        build g [_,a,ma,mi] = Info{generationRun=g, avgFit=(read a), maxFit=(read ma), minFit=(read mi)}
        build _ x = error (show x)

filterScores :: [GenerationRunInfo] -> [GenerationRunInfo]
filterScores = filter ((== 1) . (flip mod testVectorChangeFreq) . generationRun)

getAverages :: [GenerationRunInfo] -> [Float]
getAverages info = map makeAvg [avgFit, maxFit, minFit]
  where makeAvg :: (GenerationRunInfo -> Float) -> Float
        makeAvg f = average $ map f info

getAvgTable :: Int -> Scores -> GenerationRunInfo
getAvgTable run scores = 
  formatResults . getAverages $ filterScores scores
    where formatResults :: [Float] -> GenerationRunInfo
          formatResults [avgF, maxF, minF] = Info{generationRun=run, avgFit=avgF, maxFit=maxF, minFit=minF}
          formatResults x = error $ concatMap show x

getStableGen :: Scores -> Int
getStableGen scores =
  if null stable then -1 else generationRun $ head stable 
  where stable = until allBigger dropSmall $ filterScores scores
        allBigger = all ((>= stableGenTheshold) . maxFit) . take 50
        dropSmall = dropWhile (\info -> maxFit info < stableGenTheshold) . 
                    dropWhile (\info -> maxFit info >= stableGenTheshold)

checkErrors :: String -> Int -> IO [String]
checkErrors path run = do
  fileExist logFile >>= \exists -> if exists 
    then readFile logFile >>= return . foldl findSuspects [] . lines
    else return . (:[]) $ "Log file for run " ++ show run ++ " does not exist!"
  where logFile = path ++ show run ++ [pathSeparator] ++ logFilename
        findSuspects errors line = if any (flip isInfixOf line) logSuspects
          then ("Suspect in run " ++ show run ++ ": " ++ line) : errors
          else errors

processFolder :: String -> IO ()
processFolder path = do
  fileExist path >>= \exists -> if exists 
    then do
      putStrLn $ "=> Processing folder " ++ path
      stats <- foldM (foldRunStats path) emptyRunStats [1..maxRuns]
      writeFile resultsFile $ unlines [resultsHeader, formatErrors (errors stats),
--                   formatStableGen (stableGen stats), 
                   formatTable (avgTable stats)]
    else putStrLn $ "Folder " ++ path ++ " does not exist!"
  where resultsFile = path ++ resultsFilename
        resultsHeader = "EACirc - results for job " ++ job ++ "\n"
        job = reverse . takeWhile (not . (==pathSeparator)) . tail $ reverse path
        formatErrors stats = if null stats then "No error suspercts found.\n"
                       else "Error suspects in this job:\n" ++ unlines stats
        formatTable stats = unlines $ tableHeader : map formatAvgRow stats ++ [avgAvgRow stats]
        formatAvgRow row = intercalate "\t" $ (show $ generationRun row) : map (show . flip ($) row) [avgFit,maxFit,minFit]
        avgAvgRow stats = "avg:" ++ concatMap (('\t':) . show . average . flip ($) stats . map) [avgFit,maxFit,minFit]
        tableHeader = "note: Following table displays averages for all runs.\n"++
                      "run\taverage\tmaximum\tminimum"
        formatStableGen stats = if noStable stats then "No stable generation found.\n"
                                else stableGenHeader ++ (unwords $ map (\x -> iff (x == -1) "-" (show x)) stats) ++
                                     "\naverage: " ++ (show . average . map fromIntegral $ filter (>0) stats) ++ "\n"
        noStable = null . filter (>0)
        stableGenHeader = "Stable generations (after test vector change, "++
                          "fit does not drop below " ++ show stableGenTheshold ++ " for " ++ 
                          show (50 * testVectorChangeFreq) ++ " gens):\n"

average :: Fractional a => [a] -> a
average xs = sum xs / (fromIntegral $ length xs)

iff :: Bool -> a -> a -> a
iff c t e = if c then t else e

help :: String
help = "\nusage: get_scores <experiemnt-folder> [<experiment-folder> ...]\n" ++
         "e. g.  get_scores ../../boinc/_processed/*/\n\n" ++
       " - takes lines from GAlib score file after test set change\n" ++
       " - computes avg average, avg maximum and avg minimum fitness values\n" ++
       " - reads log file and filters error suspects (\"" ++ intercalate "\", \"" logSuspects ++ "\")\n" ++
       " - note: experiment folder must end with path separator, i.e '/'"

settings :: String
settings = "EACirc scores processor settings:\n" ++
           "score filename:          " ++ scoreFilename ++ "\n" ++
           "log filename:            " ++ logFilename ++ "\n" ++
           "results filename:        " ++ resultsFilename ++ "\n" ++
           "number of runs:          " ++ show maxRuns ++ "\n" ++
           "test vector change freq: " ++ show testVectorChangeFreq

main :: IO ()
main = do
  cliArgs <- getArgs
  putStrLn settings
  if length cliArgs == 0 then putStrLn help else mapM_ processFolder cliArgs
