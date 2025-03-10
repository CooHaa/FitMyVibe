# 4300-Flask-Template-SQL

## Contents

- [Summary](#summary)
- [Deploying on the server](#deploying-on-the-server )
- [Running Locally](#running-locally)
- [Uploading Large Files](#uploading-large-files)
- [MySQL functionality](#mysql-functionality)
- [Debugging Some Basic Errors](#debugging-some-basic-errors)
- [Virtual Environments and Dependency Tracking](#virtual-environments-and-dependency-tracking)
- [Troubleshooting](#troubleshooting)
- [General comments from the author](#general-comments-from-the-author)

## Summary

This is a template for **"CS/INFO 4300 class at Cornell University"**

You will use this template to directly deploy your Flask code on the project server.

After you follow the steps below, you should have set up a public address dedicated to your team's project at (for the moment) a template app will be running.  In future milestones you will be updating the code to replace that template with your very own app.


## Deploying on the server 

For the initial deployment, only one member of your team needs to follow the steps below.


### Step 0: Forking or Cloning this template

- You should make a FORK or CLONE of this repository on (regular/your public) GitHub, make sure that your repository is PUBLIC.  Keep in mind that other students will be able to see your repository.

### Step 1: Login to the deployment dashboard

- Login to the dashboard at https://4300showcase.infosci.cornell.edu/login using your provided account name and password. Each team was provided with joint account in CMS comments to assignment P01; since you are sharing an account, any changes made by one teammate will reflect for everyone in the team.
- When you login, ensure your dashboard has the following data filled from the image below (check the black arrows only)
  - The GitHub URL field will not be filled in for you, so you should add in the URL of your forked repository.

![SQL Template](https://github.com/CornellNLP/4300-Flask-Template-SQL/assets/42887079/94811cbd-549d-4091-b208-8d8eca8c6ee7)


### Step 2: Understanding the interface

- **CLONE**: First time clone from GitHub onto the server, this is to load your files on the server. In future, when you push updates, clone will re-downloaded your new files onto the server. It is imperative that you re-clone before building.
- **BUILD**: Will re-clone and build everything from your GitHub repo, and only from the master/main branch. This is a hard reset, however, your data will be preserved. This includes all data from your database and tables. 
- **DESTROY**: Will destroy all your containers as well as remove any data associated with them. Useful for fresh boot from scratch
- **Container Information Table**: Will show you the status of all your containers. This should tell you if they are on/off. Generally, this information is just useful for debugging and for checking any port errors or mismatches, although mostly just useful for TAs.
- **Logs**: Should give you an idea of what went wrong during deployment. This of course will not tell you if something is broken during build time, but only what happened when your code was deployed. 

### Step 3: Test deployment

- On the dashboard, in the provided search bar, add the URL of your forked repository
- Click the **clone** button and wait for a bit till you get a confirmation
- Click **build**, and wait for a minute. If all goes successfully, hitting the refresh button on the Container Information table and the logs tab will show the created data from the service. If this doesn't work, logout and log back in.
- Now, clicking on the URL button should lead you to a simple episode-searching app
- If it doesn't load initially, give it a few seconds and reload.
- This should be the screen you see. Test it out

![image](https://user-images.githubusercontent.com/55399795/224594465-e317dd02-7519-4fd7-aaca-32457650ce36.png)

## Running locally

- This is not formally a requirement of P01.  This is to help you test and develop your app locally; we recommend each member of the team to try this out. 
- Ensure that you have Python version 3.10 or above installed on your machine (ideally in a virtual environment). Some of the libraries and code used in the template, as well as on the server end, are only compatible with Python versions 3.10 and above.
- Make sure to pre-install MySQL Server and MySQL Workbench on your machine.
  
### Step 1: Set up MySQL
You will need to install MySQL. Here are two tutorials that could help you with the process:
- For Windows users: https://blog.devart.com/how-to-install-mysql-on-windows-using-mysql-installer.html
  - Select CUSTOM installation and remove any Visual Studio dependencies
- For Mac users: Preferably use homebrew. Your default password will be empty (""). If not, follow this https://www.geeksforgeeks.org/how-to-install-mysql-on-macos/
- For Linux users: https://www.geeksforgeeks.org/how-to-install-mysql-on-linux/


You may choose to install MySQL in an alternative method such as brew, but you will need to figure it out on your own. Regardless, make sure you write down the root password you set during the installation process. You will need it later.

We advise against using another database system such as PostgreSQL. Note that our project server uses MySQL. The different flavors of SQL may cause your app to fail on our server while working perfectly fine on yours.

### Step 2: Set up a virtual environment
Create a virtual environment in Python. You may continue using the one you setup for assignment if necessary. To review how to set up a virtual environment and activate it, refer to A0 assignment writeup.

Run `python -m venv <virtual_env_name>` in your project directory to create a new virtual environment, remember to change <virtual_env_name> to your preferred environment name.

### Step 3: Install dependencies
You need to install dependencies by running `python -m pip install -r requirements.txt` in the backend folder.

### Step 4: Connection to MySQL

## NOTE: Post bugfix: 

Make sure your MySQL server is running, then in app.py, change the SQL credentials to match your local MySQL credentials.

```flask run --host=0.0.0.0 --port=5000```

## Uploading Large Files
- When your dataset is ready, it should be of the form of an SQL file of 128MB or less.
  - 128MB is negotiable, based on your dataset requirements
  - SQL files can be exported using ```mysqldump -u root -p -d <database name> > dump.sql```
    -  NOTE: Don't just copy paste the command and expect it to work, as it is based on your OS, path variables, installations etc. If you get stuck, feel free to post on ED, although Stackoverflow will likely have a solution anyway
- Click "Choose file" and upload your file. Hit the upload button to send it to your project
- The files are chunked. Any interruption either on the network or client end will require a full file re-upload so be careful
  - In the event your file does not get consistently uploaded due to network issues or takes too long (it really shouldn't) you may request a manual upload
- This SQL file that you upload will always replace your **init.sql** file. This means that when you build your project, this file will be automatically imported into your Database and be available to use. Remember to tweak the **app.py** file to include your new database name.

## MySQL functionality

- Firstly, only use MySQL. No Postgres, no MongoDB and no SQLite
  - If you decide to use these, the server can still build them and deploy them with no problem, but you will be responsible for making it work
- A helper class called **MySQLDatabaseHandler.py** has been provided.
  - This class abstracts the process of creating and managing the database, the engine and the connections.
  - It also abstracts the process of querying the database.
  - The query_executor method will handle any non-select queries, like INSERT, UPDATE, DELETE etc. This is useful for modifying the DB as required
  - The query_selector method will return any SELECT queries made on the DB.
  - Preferably, you will not use any of the above two methods and will instead just implement your own in a more efficient way, but these functions have been provided just as an example, or as support for those who may not be comfortable with SQLAlchemy. If you are comfortable with SQLAlchemy, feel free to write the methods using the ORM framework and supported methods.
  - **NOTE: Do not modify the other methods besides the two mentioned. You can add new ones, and override the above two methods, but do not delete or modify the connection class**
- A few things to keep in mind:
  - If your database does not exist, it should automatically be created by the script (if it doesn't, post it up on ED)
  - Your authentication details for the DB are fixed along with the initial DB. 
   - Do not change these params unless you're aware of how the docker-compose file works.
- The **init.sql** file is special, in that as the name suggests, it's your de-facto DB. It will always be built before your service is ready to run, and is helpful in storing pre-existing data, like test users, some configs and anything else that you may want at run-time.
  - It has the ability to detect its environment, and will adapt based on whether you have deployed it on the server or not
  - When running locally, it will be loaded to your local database without any import commands required, and will be re-built each time
  - When deployed on the server however, it will only be run once at the start of deployment. Any changes made to the DB from here on will be permanent, unless destroyed.

## Debugging Some Basic Errors
- After the build, wait a few seconds as the server will still be loading, especially for larger applications with a lot of setup
- **Do not change the Dockerfiles without permission**
- Sometimes, if a deployment doesn't work, you can try logging out and back in to see if it works
- Alternatively, checking the console will tell you what error it is. If it's a 401, then logging in and out should fix it. 
- If it isn't a 401, first try checking the logs or container status. Check if the containers are alive or not, which could cause issues. If the containers are down, try stopping and starting them. If that does not work, you can report it on ED.
- If data isn't important, destroying and then cloning and re-building containers will usually fix the issue (assuming there's no logical error)

## Virtual Environments and Dependency Tracking
- It's essential to avoid uploading your virtual environments, as they can significantly inflate the size of your project. Large repositories will lead to issues during cloning, especially when memory limits are crossed (Limit – 2GB). 
To prevent your virtual environment from being tracked and uploaded to GitHub, follow these steps:
1. **Exclude Virtual Environment**
   - Navigate to your project's root directory and locate the `.gitignore` file. 
   - Add the name of your virtual environment directory to this file in the following format: `<virtual_environment_name>/`. This step ensures that Git ignores the virtual environment folder during commits.

2. **Remove Previously Committed Virtual Environment**
   - If you've already committed your virtual environment to the repository, you can remove it from the remote repository by using Git commands to untrack and delete it. You will find resources online to do so.
Afterward, ensure to follow step 1 to prevent future tracking of virtual environment.

3. **Managing Dependencies**
    - Add all the new libraries you downloaded using pip install for your project to the existing `requirements.txt` file. To do so,
    - Navigate to your project backend directory and run the command `pip freeze > requirements.txt`. This command will create or overwrite the `requirements.txt` file with a list of installed packages and their versions. 
    - Our server will use your project’s `requirements.txt` file to install all required packages, ensuring that your project runs seamlessly.

## Troubleshooting

The attached google document includes a compilation of frequent issues encountered by students across various project stages, detailing whether these issues have been resolved and the solutions that were effective. We will continue to update this list with new information.

Link: https://docs.google.com/document/d/1sF2zsubii_SYJLfZN02UB9FvtH1iLmi9xd-X4wbpbo8

## General comments from the author
- Since this project was made in the span of a few weeks, it is very likely things will break from time to time. If things break, you can send an email through the course email or post to ED first.
- If you would like to see stuff added to the dashboard you can send an email through the course email and prefix the title with FEATURE REQUEST
- You can also email regarding any questions relating to the service itself. If you think things can be improved or some better logic can be implemented for certain portions, or even just want to know more about the project then feel free to do so.

