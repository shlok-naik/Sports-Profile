"""
Database layer for CombineAI scout athlete storage
"""

import mysql.connector


def get_connection():

    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="eagle:gold",
        database="combineai"
    )

def create_tables():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS athletes (

        id INT PRIMARY KEY,

        name VARCHAR(100),

        sport VARCHAR(100),

        running INT,

        high_knees INT,

        jump FLOAT,

        pushups INT,

        plank INT

    )
    """)

    conn.commit()

    cursor.close()
    conn.close()


def save_athlete(data):

    conn = get_connection()

    cursor = conn.cursor()


    sql = """
    INSERT INTO athletes
    (
        id,
        name,
        sport,
        running,
        high_knees,
        jump,
        pushups,
        plank
    )

    VALUES
    (
        %s,%s,%s,%s,%s,%s,%s,%s
    )
    """


    values = (

        data["id"],
        data["name"],
        data["sport"],
        data["running"],
        data["high_knees"],
        data["jump"],
        data["pushups"],
        data["plank"]

    )


    cursor.execute(sql, values)

    conn.commit()

    cursor.close()
    conn.close()




def get_athletes():

    conn = get_connection()

    cursor = conn.cursor(dictionary=True)


    cursor.execute(
        "SELECT * FROM athletes"
    )


    athletes = cursor.fetchall()


    cursor.close()
    conn.close()


    return athletes




def get_athlete_count():

    conn = get_connection()

    cursor = conn.cursor()


    cursor.execute(
        "SELECT COUNT(*) FROM athletes"
    )


    count = cursor.fetchone()[0]


    cursor.close()
    conn.close()


    return count
