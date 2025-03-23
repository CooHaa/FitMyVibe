import os
import sqlalchemy as db

class MySQLDatabaseHandler(object):
    
    IS_DOCKER = True if 'DB_NAME' in os.environ else False

    def __init__(self,MYSQL_USER,MYSQL_USER_PASSWORD,MYSQL_PORT,MYSQL_DATABASE,MYSQL_HOST = "localhost"):
        
        self.MYSQL_HOST = os.environ['DB_NAME'] if MySQLDatabaseHandler.IS_DOCKER else MYSQL_HOST
        self.MYSQL_USER = "admin" if MySQLDatabaseHandler.IS_DOCKER else MYSQL_USER
        self.MYSQL_USER_PASSWORD = "admin" if MySQLDatabaseHandler.IS_DOCKER else MYSQL_USER_PASSWORD
        self.MYSQL_PORT = 3306 if MySQLDatabaseHandler.IS_DOCKER else MYSQL_PORT
        self.MYSQL_DATABASE = "kardashiandb" if MySQLDatabaseHandler.IS_DOCKER else MYSQL_DATABASE
        self.engine = self.validate_connection()

    def validate_connection(self):
        print(f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_USER_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}")
        return db.create_engine(f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_USER_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}")

    def lease_connection(self):
        return self.engine.connect()
    
    def query_executor(self,query):
        conn = self.lease_connection()
        if type(query) == list:
            for i in query:
                conn.execute(db.text(i))
        else:
            conn.execute(db.text(query))
        

    def query_selector(self,query):
        conn = self.lease_connection()
        data = conn.execute(query)
        return data

    def load_file_into_db(self,file_path  = None):
        if MySQLDatabaseHandler.IS_DOCKER:
            return
        if file_path is None:
            file_path = os.path.join(os.environ['ROOT_PATH'],'init.sql')
        sql_file = open(file_path,"r")
        sql_file_data = list(filter(lambda x:x != '',sql_file.read().split(";\n")))
        self.query_executor(sql_file_data)
        sql_file.close()

