
This is a tool to generate a dataset of short Chiense stories using GPT3.5. It is inspired by Tiny Stories dataset, but instead of millions of rows, I only generated a few thousands stories. The tool and the dataset was created as a learning exercise for using GPT API to generate training data for a potential language model idea. You can preview the result data set [here](https://huggingface.co/datasets/joe-chiu/TinyChineseStories).

I created these stories by first using ChatGPT to generate a list of male and female character names, a list of genre and one sentence story themes and a list of story starters (similar to "Once upon a time"). Later, I use GPT3.5 chat completion API to generate short stories given the 3 constraints: genre and theme and sentence starter. And the stories were generated in the batch of 3. So every 3 stories would share the exact same parameters.

<img width="748" alt="image" src="https://github.com/joe-chiu/TinyChineseStories/assets/14063642/b050c63b-7a3a-43da-90f6-5e11813c50d7">
