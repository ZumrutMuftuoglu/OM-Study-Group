Hello! Here are the steps you can follow to publish a blog post on the Study Group blogging website: https://zumrutmuftuoglu.github.io/OM-Study-Group/

1. Fork the github repository, and then clone the fork to your local system. 

Please check out this tutorial if you are unfamiliar with git: [First steps with git: clone, add, commit, push Intro version control git](https://www.earthdatascience.org/workshops/intro-version-control-git/basic-git-commands/).

Please see this tutorial if you haven't forked a git repo before or made a pull request: [How To: Fork a GitHub Repository & Submit a Pull Request](https://jarv.is/notes/how-to-pull-request-fork-github/)

2. Navigate to the /docs/_posts folder, and make a copy of 2020-04-19-first-post.md. 

3. Rename the copy of 2020-04-19-first-post.md to include the date you are authoring the article and a short name for your blog post in the specified format.

4. Open this file (you just renamed) using a text editor. Change the following:

* Change title to the title of your blog post
* Change author to your name in undercase
* Change categories to a relevant tag for your article (e.g. differential-privacy, or federated-learning)
* Change image to the filepath for the relevant image for your article. Before you do this, please make sure to navigate to the assets/images/ directory, and add your .JPG or .PNG File there. Please make sure the image dimensions are around 750 x 500.

* Below the second --- line, replace the dummy text "Hello, this is our first post on the website!" with the content of your blog post. 

For the blog posts, we are working with Markdown files. See this guide by Github if you're unfamiliar with how to write content in Markdown files: [Mastering Markdown](https://guides.github.com/features/mastering-markdown/)

For more advanced "special effects" in your post: If you'd like to add in special effects to your articles, like Spoilers, check out the [Mediumish Jekyll theme website](https://wowthemesnet.github.io/mediumish-theme-jekyll/). You can also check out docs/_posts/example_featured_post2018-01-11-quick-start-guide.md in our repo for an example of how to use spoilers.

5. Once you're finished writing your blog post, head over to docs/config.yaml. Copy over one of the authors sections (e.g. Ria or John), and edit them with your information. You can add in a picture of yourself if you'd like, and a link to your websites and your email.

Note: If you're interested in seeing what your blog post would look like before making the pull request to the main repo: In your forked repo, go to the Settings tab, scroll down to the GitHub Pages section, and enable the blog from /docs option - you should then be able to see your own blog post with the link provided. 

6. Now that you've finished writing your post and filled in your author information, it's time to make a pull request to publish your post! git add, commit, and push your changes to your fork repo and then submit your pull request (Github should automatically suggest this-check the tutorial above for the detailed instructions if stuck). The creator/contributors to the repo will then review your pull request, and merge it if everything is ok! We will fhen notify you that your pull request has been merged, so you can then check to see that your article is up on the page.


If you're having any trouble with these steps (e.g. your changes aren't showing up), please make a post in the #sg-om-explorers channel, and we'll help you fix the issue!

Happy writing!
