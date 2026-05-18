CREATE TABLE IF NOT EXISTS loan_predict_train (
    loan_number INT AUTO_INCREMENT PRIMARY KEY ,
    loan_id VARCHAR(20) UNIQUE NOT NULL,
    user_id INT NOT NULL,

    gender VARCHAR(10),
    married VARCHAR(10),
    dependents VARCHAR(10),
    education VARCHAR(20),
    self_employed VARCHAR(10),

    applicant_income FLOAT,
    coapplicant_income FLOAT,
    loan_amount FLOAT,
    loan_amount_term FLOAT,
    credit_history FLOAT,
    property_area VARCHAR(20),

    prediction VARCHAR(20),
    confidence FLOAT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id) REFERENCES users(id)
);