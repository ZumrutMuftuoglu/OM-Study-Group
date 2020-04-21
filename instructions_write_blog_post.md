Hello! Here are the steps you can follow to publish a blog post on the Study Group blogging website: https://zumrutmuftuoglu.github.io/OM-Study-Group/

1. Clone the github repository to your local system. Please check out this tutorial if you are unfamiliar with git: [First steps with git: clone, add, commit, push Intro version control git](https://www.earthdatascience.org/workshops/intro-version-control-git/basic-git-commands/).

2. Navigate to the /docs/_posts folder, and make a copy of 2020-04-19-first-post.md. 

3. Rename the copy of 2020-04-19-first-post.md to include the date you are authoring the article and a short name for your blog post in the specified format.

4. Open this file (you just renamed) using a text editor. Change the following:

* Change title to the title of your blog post
* Change author to your name in undercase
* Change categories to a relevant tag for your article (e.g. differential-privacy, or federated-learning)
* Change image to the filepath for the relevant image for your article. Before you do this, please make sure to navigate to the assets/images/ directory, and add your .JPG or .PNG File there.
* Below the second --- line, replace the dummy text "Hello, this is our first post on the website!" with the content of your blog post. 

For the blog posts, we are working with Markdown files. See this guide by Github if you're unfamiliar with how to write content in Markdown files: [Mastering Markdown](https://guides.github.com/features/mastering-markdown/)


For more advanced "special effects" in your post: If you'd like to add in special effects to your articles, like Spoilers, check out the [Mediumish Jekyll theme website](https://wowthemesnet.github.io/mediumish-theme-jekyll/). You can also check out docs/_posts/example_featured_post2018-01-11-quick-start-guide.md in our repo for an example of how to use spoilers.

5. Once you're finished writing your blog post, head over to docs/config.yaml. Copy over one of the authors sections (e.g. Ria or John), and edit them with your information. You can add in a picture of yourself if you'd like, and a link to your websites and your email.

6. Now that you've finished writing your post and filled in your author information, it's time to publish your post! git add, commit, and push your changes to the repo. Wait for a few minutes, and then check to see tha your article is up on the page.

If you're having any trouble with these steps (e.g. you're changes aren't showing up), please make a post in the #sg-om-explorers channel, and we'll help you fix the issue!

Happy writing!
