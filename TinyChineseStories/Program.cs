using System.Collections.ObjectModel;
using System.Text.RegularExpressions;
using Ckeisc.OpenAi;

public class Program
{
    static string? openAiApiKey = Environment.GetEnvironmentVariable("OPENAI_APIKEY");
    const string ChineseConversionFile = "hcutf8.txt";
    // how many stories to generate in this run, may exceed this limit as we get stories in batch of 3
    const int StoryGenLimit = 100;
    const int WorkerThreadCount = 8;

    static Dictionary<char, char> simplifiedToTraditionalMap = GetChineseLookupMap();

    const string storyTemplate = 
        "創作三篇 {0}的極短篇故事, 每一個故事長度四百字 " + 
        "用兩個人物 男性BOY1跟女性GIRL1, 故事主題是 '{1}', " + 
        "每個故事用 '{2}' 這個句子開頭 不要重複故事主題 enclose each story in <story> and </story> tags";

    const string OutputFilename = "stories.csv";
    private static object outputLock = new();
    private static object listLock = new();

    // load data files
    private static ReadOnlyCollection<string> genreList = ReadData("genre.txt").AsReadOnly();
    private static ReadOnlyCollection<string> themeList = ReadData("theme.txt").AsReadOnly();
    private static ReadOnlyCollection<string> starterList = ReadData("starter.txt").AsReadOnly();
    private static ReadOnlyCollection<string> boyNameList = ReadData("name-boy1.txt").AsReadOnly();
    private static ReadOnlyCollection<string> girlNameList = ReadData("name-girl1.txt").AsReadOnly();

    private static bool done = false;
    private static int count = 0;
    private static SemaphoreSlim throttler = new SemaphoreSlim(initialCount: WorkerThreadCount);
    private static List<Task> pendingTasks = new();

    public static async Task Main()
    {
        if (string.IsNullOrWhiteSpace(openAiApiKey))
        {
            Console.WriteLine("Please set OpenAI API key in the OPENAI_APIKEY env variable.");
            return;
        }

        OpenAiClient client = new OpenAiClient(openAiApiKey);
        InitializeOutput(OutputFilename);

        while (!done)
        {
            await throttler.WaitAsync();
            
            Task task = Task.Run(async () => {
                Console.WriteLine($" * Work starting... TID:{Thread.CurrentThread.ManagedThreadId}");

                Random rand = new Random();
                string starter = starterList[rand.Next(starterList.Count)];
                string genre = genreList[rand.Next(genreList.Count)];
                string theme = themeList[rand.Next(themeList.Count)];

                string prompt = string.Format(storyTemplate, genre, theme, starter);

                ChatMessage[] chatMessages = CreateChatMessage(prompt);

                try
                {
                    // make sure there is enough room in token limit, the limit 4097 includes promopt token count
                    ChatCompletionResponse response = 
                        await client.CreateChat(chatMessages, ChatModels.Gpt3_5Turbo, 3800);

                    string[]? stories = ExtractStories(response, 3);

                    if (stories != null && stories[0] != null)
                    {
                        foreach(string story in stories)
                        {
                            // swap in boy1 and girl1 name and write good entries to DB
                            string boyName = boyNameList[rand.Next(boyNameList.Count)];
                            string girlName = girlNameList[rand.Next(girlNameList.Count)];
                            string processedStory = story.Replace(
                                "BOY1", boyName, StringComparison.InvariantCultureIgnoreCase);
                            processedStory = processedStory.Replace(
                                "GIRL1", girlName, StringComparison.InvariantCultureIgnoreCase);
                            // need to remove comma to avoid breaking CSV files
                            processedStory = processedStory.Replace(
                                ",", "，");

                            // alternate cleaning method
                            if (processedStory.Contains(","))
                            {
                                processedStory = $"\"{processedStory}\"";
                            }

                            // normalize to traditional Chinese, if text already in CHT
                            // this does not break it
                            processedStory = ChsToCht(processedStory);

                            AddStory(boyName, girlName, theme, genre, processedStory);
                            Console.WriteLine($"Story added: {genre}|{theme}");
                            count++;

                            if (count >= StoryGenLimit)
                            {
                                done = true;
                            }
                        }
                    }
                    else
                    {
                        Console.WriteLine($"[Warning] failed to parse response: {response.Choices[0].Message.Content}");
                    }

                    Console.WriteLine($" * Work done... TID:{Thread.CurrentThread.ManagedThreadId}");
                }
                catch (Exception e)
                {
                    Console.WriteLine($"[Warning] task failed: {e.Message}");
                }
                finally
                {
                    throttler.Release();
                }
            });

            Task dontCare = task.ContinueWith((Task t) => {
                lock(listLock)
                {
                    pendingTasks.Remove(t);
                }
            });

            lock (listLock)
            {
                pendingTasks.Add(task);
            }
        }

        await Task.WhenAll(pendingTasks.ToArray());
    }

    static void AddStory(
        string boy, 
        string girl, 
        string theme,
        string genre,
        string story)
    {
        lock(outputLock)
        {
            string line = $"{boy}|{girl},{genre},{theme},{story}{Environment.NewLine}";
            File.AppendAllText(OutputFilename, line);
        }
    }

    static void InitializeOutput(string filename)
    {
        if (!File.Exists(filename))
        {
            Console.WriteLine($"Creating {filename}...");

            using StreamWriter writer = new StreamWriter(File.Create(filename));
            // write out column names
            writer.WriteLine("names, genre, theme, story");
            writer.Close();
        }
        else
        {
            Console.WriteLine($"Appending {filename}...");
            using StreamReader reader = File.OpenText(filename);
            int lineCount = 0;
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                lineCount++;
            }
            reader.Close();
            // this includes header and empty lines (if any), not trying to parse lines
            Console.WriteLine($"{lineCount} lines in {filename}...");
        }
    }

    static List<string> ReadData(string file)
    {
        List<string> output = File.ReadAllLines(file).Where(
            line => !line.TrimStart().StartsWith("#") && 
            !string.IsNullOrWhiteSpace(line)).ToList();
        return output;
    }

    static ChatMessage[] CreateChatMessage(string prompt)
    {
        ChatMessage[] userChatPrompt = new[]
        {
            new ChatMessage()
            {
                Role = ChatRole.User,
                Content = prompt
            }
        };        
        return userChatPrompt;
    }

    static string[]? ExtractStories(
        ChatCompletionResponse response, 
        int expectedStories)
    {
        string[] output = new string[expectedStories];

        if (response.Choices[0].FinishReason != FinishReason.Stop)
        {
            // this response did not finish cleanly
            return null;
        }

        string content = response.Choices[0].Message.Content;

        Regex stories = new Regex("<story>([^<]+)</story>");
        MatchCollection matches = stories.Matches(content);

        if (matches.Count == expectedStories)
        {
            for(int i=0; i<expectedStories; i++)
            {
                Match match = matches[i];
                output[i] = match.Groups[1].Value.Trim().Replace("\n", "");
            }
        }

        return output;
    }

    static Dictionary<char, char> GetChineseLookupMap()
    {
        Dictionary<char, char> simplifiedToTraditionalMap = new();
        using StreamReader reader = File.OpenText(ChineseConversionFile);
        string? line;
        while ((line = reader.ReadLine()) != null)
        {
            line = line.Trim();
            // comment / empty
            if (line.StartsWith("#") || string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            if (line.Length >= 2)
            {
                char simplifiedChar = line[0];
                char traditionalChar = line[1];

                if (simplifiedChar == traditionalChar)
                {
                    continue;
                }

                if (simplifiedToTraditionalMap.ContainsKey(simplifiedChar))
                {
                    Console.WriteLine($"{simplifiedChar} already mapped to {simplifiedToTraditionalMap[simplifiedChar]}");
                    continue;
                }

                simplifiedToTraditionalMap[simplifiedChar] = traditionalChar;
            }
        }
        reader.Close();
        return simplifiedToTraditionalMap;
    }

    static void CleanFile()
    {
        using StreamReader reader = File.OpenText(OutputFilename);
        string? line;
        while ((line = reader.ReadLine()) != null)
        {
            // remove lines with tokens that failed to be swapped in earlier version
            if (line.Contains("boy1", StringComparison.InvariantCultureIgnoreCase) ||
                line.Contains("girl1", StringComparison.InvariantCultureIgnoreCase))
            {
                continue;   
            }

            File.AppendAllText("cleaned.txt", line + Environment.NewLine);
        }
        reader.Close();
    }

    static void CreateChtStoryOnlyFile()
    {
        using StreamReader reader = File.OpenText(OutputFilename);
        string? line;
        bool firstLine = true;
        while ((line = reader.ReadLine()) != null)
        {
            // skip header
            if (firstLine)
            {
                firstLine = false;
                continue;
            }

            string[] parts = line.Split(',', 4, StringSplitOptions.TrimEntries);
            string chtString = ChsToCht(parts[3]);

            File.AppendAllText("story-only-cht.txt", 
                chtString + Environment.NewLine + Environment.NewLine);
        }
        reader.Close();
    }

    static string ChsToCht(string chsString)
    {
        List<char> chtCharList = chsString.ToList().ConvertAll<char>(
            (char input) => simplifiedToTraditionalMap.ContainsKey(input) ?
                simplifiedToTraditionalMap[input] : input);
        string chtString = string.Join("", chtCharList);
        return chtString;
    }
}