import os
from tkinter import * # tkinter의 모든 기능 import
import pandas as pd

window = Tk()  # Tk() 생성자로 window 생성 (부모 컨테이너)

frame1 = Frame(window)  # 프레임 = 여러 위젯을 묶는 위젯. 컨테이너 역할.

label_subtitle = Label(frame1, text=" 해당 선수의 2019년 예상 OPS는 다음과 같습니다. ")

os.chdir("/home/daeho/다운로드/drive-download-20191022T014909Z-001")

ops_data = pd.read_csv("submission01.csv")

name = StringVar()

text = Text(window)

class Widget: # class 생성.

    def __init__(self):

        # 기하 관리자. 위젯 배치를 함. (레이블, 버튼 등)
        window.title("2019 KBO 타자들의 OPS를 예측해보자.")  # window 창 뜰때 메인 이름

        frame1.pack() # 프레임으로 묶인 위젯들(label_title, entryName, btGetName, btCancel)을 부모 위젯(window)에 패킹함.

        label_title = Label(frame1, text="   2019년 OPS가 궁금한 타자의 이름을 입력해주세요.   ")
        entryName = Entry(frame1, textvariable=name) # 이름을 문자열 형태로 입력받음.

        label_main = Label(window, textvariable=name) # 입력받은 문자열(이름)을 위젯에 출력함.

        # 버튼은 이벤트 엔트리의 이름을 가져와야한다.
        # command = 함수 ---> 해당 이벤트가 발생하면, 해당 함수를 실행.
        btGetName = Button(frame1, text="SEARCH", command=self.processSearch)
        btCancel = Button(frame1, text="Cancel", command=self.processCancel)
        btClear = Button(frame1, text="다른 선수 검색!", command=self.processClear)

        # 기하관리자 ~~~> grid() / pack()
        # grid(행, 열) ~> 지정된 행, 열에 위젯 배치.
        # pack() ~> 위젯을 부모 위젯(여기서는 window)에 모두 패킹(붙임), 불 필요 공간 사라짐.
        label_title.grid(row=1, column=1)
        label_main.pack()

        entryName.grid(row=1, column=2)

        btGetName.grid(row=1, column=3)
        btClear.grid(row=1, column=4)
        btCancel.grid(row=1, column=5)

        # 이벤트 루프. 이벤트 발생까지 대기후, 발생하면, 기능을 수행.
        window.mainloop()
        # 사용자가 무엇인가를 작업 할 때까지 계속 루프를 돌며 대기하고, 작업이 들어오면 처리.
        # 모든 GUI 프로그램의 끝에는 반드시 메인 루프를 호출해야 GUI 프로그램이 지속적으로 대기한다.
        # 이벤트 루프 시작 -> 이벤트 감지 및 처리 -> 메인 창 종료 -> 사용자 종료? -'NO'-> 다시 이벤트 루프 시작 -> ...
        # 이벤트 : 마우스 클릭, 키보드 입력, 타이머 등

    # 이벤트 핸들러 정의.
    def processSearch(self):
        player_result = ops_data[ops_data["batter_name"] == name.get()]
        print(" 해당 타자의 2019년 예상 OPS는 다음과 같습니다. ")
        print("=========================================")
        print(player_result)
        text.insert(END, player_result)
        text.pack()

    def processClear(self):
        text.delete(1.0, END)

    def processCancel(self):
        print("검색을 취소합니다!")
        window.quit()

Widget() # 클래스 생성자 호출. 코드 실행



