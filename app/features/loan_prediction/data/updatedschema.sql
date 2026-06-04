-- MySQL Workbench Synchronization
-- Generated: 2026-05-10 19:59
-- Model: New Model
-- Version: 1.0
-- Project: Name of the project
-- Author: Nelson

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

ALTER TABLE `railway`.`loan_predict_train` 
CHANGE COLUMN `Gender` `Gender` VARCHAR(45) NULL DEFAULT NULL ,
CHANGE COLUMN `Married` `Married` VARCHAR(45) NULL DEFAULT NULL ,
CHANGE COLUMN `Dependents` `Dependents` INT(45) NULL DEFAULT NULL ,
CHANGE COLUMN `Education` `Education` VARCHAR(45) NULL DEFAULT NULL ,
CHANGE COLUMN `Self_Employed` `Self_Employed` VARCHAR(45) NULL DEFAULT NULL ,
CHANGE COLUMN `ApplicantIncome` `ApplicantIncome` INT(11) NULL DEFAULT NULL ,
CHANGE COLUMN `CoapplicantIncome` `CoapplicantIncome` INT(11) NULL DEFAULT NULL ,
CHANGE COLUMN `LoanAmount` `LoanAmount` VARCHAR(45) NULL DEFAULT NULL ;


DELIMITER $$

USE `railway`$$
DROP TRIGGER IF EXISTS `railway`.`loan_predict_train_BEFORE_INSERT` $$

USE `railway`$$
CREATE DEFINER = CURRENT_USER TRIGGER `railway`.`loan_predict_train_BEFORE_INSERT` BEFORE INSERT ON `loan_predict_train` FOR EACH ROW
BEGIN
	DECLARE next_number INT;

    -- Get next loan number
    SELECT IFNULL(MAX(loan_number), 1000) + 1
    INTO next_number
    FROM loan_predict_train;

    -- Set loan_number
    SET NEW.loan_number = next_number;

    -- Generate formatted loan_id
    IF next_number <= 9999 THEN
        SET NEW.loan_id = CONCAT('LP00', next_number);
    ELSEIF next_number <= 99999 THEN
        SET NEW.loan_id = CONCAT('LP0', next_number);
    ELSE
        SET NEW.loan_id = CONCAT('LP', next_number);
    END IF;
END$$


DELIMITER ;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
