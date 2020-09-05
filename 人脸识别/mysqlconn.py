import pymysql
import datetime

conn=pymysql.Connect(
    host='localhost',
    port=3306,
    user='root',
    passwd='12345',
    db='hongshixing',
    charset='utf8')

cur = conn.cursor()
if not cur:
    raise Exception('数据库连接失败！')
result = {'selCount': -1,'selname':'','ScuCount': ''}
nowt = datetime.datetime.now()
def getdata(stunum):
    sSQL = "select count(*) as scount from stusignininfo where stunum='"+str(stunum)+"' and signinstatus=1 and (signintime between STR_TO_DATE('"+(nowt-datetime.timedelta(minutes=90)).strftime('%Y-%m-%d %H:%M:%S')+"','%Y-%m-%d %H:%i:%s') and STR_TO_DATE('"+nowt.strftime('%Y-%m-%d %H:%M:%S')+"','%Y-%m-%d %H:%i:%s'))"
    conn.ping(reconnect=True)
    cur.execute(sSQL)
    rs=cur.fetchone()
    result['selCount']=rs[0]

def insertdata(stunum):
    cur = conn.cursor()
    if not cur:
        raise Exception('数据库连接失败！')
    getdata(stunum)
    if(int(result['selCount']) == 0):
        insSQL = "insert stusignininfo(stunum,stuname,signinstatus,signintime) values('"+str(stunum)+"',(select studentname from studentinfo where studentnum='"+str(stunum)+"'),1,STR_TO_DATE('"+nowt.strftime('%Y-%m-%d %H:%M:%S')+"','%Y-%m-%d %H:%i:%s'))"
        result['ScuCount']=cur.execute(insSQL)
        conn.commit()
    conn.close()
    return result

