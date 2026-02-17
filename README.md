# Powerlifting Trainer's Assistant API- made by Dor Habooshe, Itay Golan and Moran Herzlinger
Link to the video- https://youtu.be/xNZTbHj9JV0

We developed a Powerlifting Trainer's Assistant designed to enhance training routines through a series of functionalities built using **FastAPI** and **SQLite**.
The project starts with a **registration function** where users can register their email, height, weight, and name.
Upon registration, the system calculates the user's BMI and stores their data along with three empty lists for tracking record weights in a **SQLite database**.
The project also includes a **search function** that allows users to retrieve their profile information by entering their email address.
If the email exists in the database, the corresponding data is displayed. 
otherwise, the system notifies the user that the profile does not exist. 
Additionally, the **record weight update function** enables users to log and update their personal bests in various powerlifting exercises, with the records stored in the database.

The core of the project is the **video processing function**, where trainers can upload videos of exercises or perform them in real time.
The system analyzes the minimum angle achieved during exercises, such as the knee angle in squats or the back angle in deadlifts, and sends this data to the **OpenAI API** for feedback on the trainee's form and tips for improvement.
Real-time exercise analysis is also supported, with the system providing immediate audio feedback when an exercise is performed correctly, counting successful repetitions, and requesting additional feedback from OpenAI.
The feedback is then sent to the trainee's email and displayed on the screen. The entire API is deployed using **Docker** on **Azure**, with continuous integration and deployment managed through **Azure Pipelines**. 

This project combines various technologies to offer a comprehensive solution for powerlifting trainers, enabling them to monitor and improve their trainees' performance through real-time feedback and personalized recommendations.
