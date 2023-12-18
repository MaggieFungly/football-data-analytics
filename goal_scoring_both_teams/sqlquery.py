import os
from pyspark.sql import SparkSession
import pandas as pd

# mysql connector
os.environ['PYSPARK_SUBMIT_ARGS'] = '--driver-class-path /home/lymf/mysql-connector-j-8.1.0.jar pyspark-shell'

spark = SparkSession.builder.appName("MySQLIntegration").getOrCreate()


# get connected to the dataset
def read_mysql_table(table_name):
    return spark.read.format("jdbc").options(
        url="jdbc:mysql://localhost:3306/ftdata",
        driver="com.mysql.jdbc.Driver",
        dbtable=table_name,
        user="debian-sys-maint",
        password="vMCs6xaR5jYNdv3u"
    ).load()


# get tables
print("connecting to database...")

match_events_2021 = read_mysql_table("match_events_2021")
match_events_1920 = read_mysql_table("match_events_1920")
match_events_1819 = read_mysql_table("match_events_1819")
matches = read_mysql_table("matches")

match_events_1819.createOrReplaceTempView('match_events_1819')
match_events_1920.createOrReplaceTempView('match_events_1920')
match_events_2021.createOrReplaceTempView('match_events_2021')
matches.createOrReplaceTempView('matches')


# query from each table
print('fetching data...')


def get_sql_query(season: str):
    sql_query = f"""
                SELECT m1.*, m2.shot_id
                FROM match_events_{season} AS m1 
                JOIN (
                    SELECT `index`, `match_id`, id AS shot_id 
                    FROM match_events_{season}
                    WHERE `type` = 'Shot' AND `index` >= 10
                ) AS m2 
                ON m1.`match_id` = m2.`match_id`
                WHERE m1.`index` BETWEEN (m2.`index` - 10) AND m2.`index` 
                ORDER BY m1.`match_id`, m2.`shot_id`, m1.`index`;
                """
    return sql_query


events_1819 = spark.sql(get_sql_query('1819')).toPandas()
events_1920 = spark.sql(get_sql_query('1920')).toPandas()
events_2021 = spark.sql(get_sql_query('2021')).toPandas()


