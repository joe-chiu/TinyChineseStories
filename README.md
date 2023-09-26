
This is a dataset of short Chiense stories generated from GPT3.5. It is inspired by Tiny Stories dataset, but instead of millions of rows, I only generated a few thousands stories. The dataset was created as a learning exercise for using GPT API to generate training data for a potential language model idea. 

I created these stories by first using ChatGPT to generate a list of male and female character names, a list of genre and one sentence story themes and a list of story starters (similar to "Once upon a time"). Later, I use GPT3.5 chat completion API to generate short stories given the 3 constraints: genre and theme and sentence starter. And the stories were generated in the batch of 3. So every 3 stories would share the exact same parameters.
