DROp DATABASE IF EXISTS rumour;
CREATE DATABASE rumor1;
USE rumor1;

CREATE TABLE `users` (
    `name` VARCHAR(225),
    `email` VARCHAR(225),
    `phonenumber` VARCHAR(225),
    `password` VARCHAR(225),
    `age` VARCHAR(225),
    `place` VARCHAR(225)
);


CREATE TABLE `feedbacks` (
    `ID` INT PRIMARY KEY AUTO_INCREMENT,
    `user_name` VARCHAR(225),
    `user_email` VARCHAR(225),
    `rating` INT,
    `comment` VARCHAR(225)
);