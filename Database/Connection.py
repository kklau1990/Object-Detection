import pymongo
import os
import xml.etree.ElementTree as ET


DB_CONFIG = f'{os.getcwd()}\\dbconfig.xml'

db_config = ET.parse(DB_CONFIG)
root = db_config.getroot()

server_name = root.find('Server')[0].text
db_name = root.find('Database')[0].text

client = pymongo.MongoClient(f'mongodb://{server_name}')

db_obj = client[db_name]

