from flask import Flask, render_template, Response
import pandas as pd
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN 

app=Flask(__name__)

detector = MTCNN()


csv_path_face = 'color_face.xlsx'
csv_path_hair = 'color_hair.xlsx'
csv_path_eye = 'color_eye.xlsx'

index = ['color', 'color_name', 'hex', 'R', 'G', 'B','R1', 'G1', 'B1']
df_f = pd.read_excel(csv_path_face, names=index, header=None)
df_h = pd.read_excel(csv_path_hair, names=index, header=None)
df_e = pd.read_excel(csv_path_eye, names=index, header=None)

#_____________________________________________________________________________________________________________________
#define RGB for face
def get_color_name_face(R,G,B):
    minimum = 1000
    for i in range(len(df_f)):
        Rd = ((df_f.loc[i,'R'])+(df_f.loc[i,'R1']))/2
        Gd = ((df_f.loc[i,'G'])+(df_f.loc[i,'G1']))/2
        Bd = ((df_f.loc[i,'B'])+(df_f.loc[i,'B1']))/2
        d = abs(R - int(Rd)) + abs(G - int(Gd)) + abs(B - int(Bd))
        if d <= minimum:
            minimum = d
            cname = df_f.loc[i, 'color_name']

    return cname
#______________________________________________________________________________________________________________________
#define RGB for hair
def get_color_name_hair(R,G,B):
    minimum = 1000
    for i in range(len(df_h)):
        Rd = ((df_h.loc[i,'R'])+(df_h.loc[i,'R1']))/2
        Gd = ((df_h.loc[i,'G'])+(df_h.loc[i,'G1']))/2
        Bd = ((df_h.loc[i,'B'])+(df_h.loc[i,'B1']))/2
        d = abs(R - int(Rd)) + abs(G - int(Gd)) + abs(B - int(Bd))
        if d <= minimum:
            minimum = d
            cname = df_h.loc[i, 'color_name']

    return cname
#_______________________________________________________________________________________________________________________________
def get_color_name_eye(R,G,B):
    minimum = 1000
    for i in range(len(df_e)):
        Rd = ((df_e.loc[i,'R'])+(df_e.loc[i,'R1']))/2
        Gd = ((df_e.loc[i,'G'])+(df_e.loc[i,'G1']))/2
        Bd = ((df_e.loc[i,'B'])+(df_e.loc[i,'B1']))/2
        d = abs(R - int(Rd)) + abs(G - int(Gd)) + abs(B - int(Bd))
        if d <= minimum:
            minimum = d
            cname = df_e.loc[i, 'color_name']

    return cname
#_______________________________________________________________________________________________________________________________

def gen_frames():
    camera  = cv2.VideoCapture(0)
	
    while True:
        # read the camera frame
        success, frame = camera.read()  
        if not success:
            break
        else: 
            result = detector.detect_faces(frame)
            try:
                bounding_box = result[0]['box'] 
                centerCoord_hair = (bounding_box[0]+(bounding_box[2]/2), (bounding_box[1]-(bounding_box[3]/2)-35)+(bounding_box[3]/2))
                forehead = ((bounding_box[0]+(bounding_box[2]/2)),(bounding_box[1]+(bounding_box[3]/6)))
                left_eye = result[0]['keypoints']['left_eye']
                right_eye = result[0]['keypoints']['right_eye']

                # Define Color For Face
                cv2.circle(frame,(int(forehead[0]),int(forehead[1])),5,(0,155,255), 2)
                y = int(forehead[0])
                x = int(forehead[1])
                (b, g, r) = frame[y,x]
                print("Face COlor: ",get_color_name_face(b,g,r))

                face_color = get_color_name_face(b,g,r)
                label =  'Face Color: %s' % get_color_name_face(b,g,r)
                p = (bounding_box[0]+(bounding_box[2]/6))
                q = (bounding_box[1]+(bounding_box[3]/2))
                cv2.putText(frame,label,(int(p),int(q)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))

#__________________________________________________________________________________________________________________________________
                # Define Color For Hair
                cv2.circle(frame,(int(centerCoord_hair[0]),int(centerCoord_hair[1])),5,(0,155,255), 2)
                y1 = int(centerCoord_hair[0])
                x1 = int(centerCoord_hair[1])
                (b1, g1, r1) = frame[y1,x1]
                print("Hair color: ", get_color_name_hair(b1,g1,r1)) 

                hair_color = get_color_name_hair(b1,g1,r1)
                label =  'Hair Color: %s' % get_color_name_hair(b1,g1,r1)
                cv2.putText(frame,label,(bounding_box[0],bounding_box[1]-int(bounding_box[3]/3)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
#_______________________________________________________________________________________________________________________________________
                # Define Color For Eye
                y2 = left_eye[0]
                x2 = left_eye[1]
                (b2, g2, r2) = frame[y2,x2] 
                print("eye color: ", get_color_name_eye(b,g,r))

                eye_color = get_color_name_eye(b,g,r)
                label =  'Eye Color: %s' % get_color_name_eye(b,g,r)
                eye_distance = np.linalg.norm(np.array(left_eye)-np.array(right_eye))
                eye_radius = eye_distance/15 # approximate
                cv2.circle(frame, left_eye, int(eye_radius), (0, 155, 255), 1)
                cv2.circle(frame, right_eye, int(eye_radius), (0, 155, 255), 1)
                cv2.putText(frame, label, (left_eye[0]-10, left_eye[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
#__________________________________________________________________________________________________________________________________________
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            except:
                print("Face not detected")
            
        # if face_color and hair_color and eye_color != '':
            # camera.release()
            # cv2.destroyAllWindows()

        if cv2.waitKey(1) &0xFF == ord('q'):
            break
    #When everything's done, release capture
    camera.release()
    cv2.destroyAllWindows()
#_____________________________________________________________________________________________________________________________________
@app.route('/')
def hello():
    return "Welcome to flask"
@app.route('/detect')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    app.run(debug=True)
