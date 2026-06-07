CREATE TABLE IF NOT EXISTS departments (
    department_id INTEGER PRIMARY KEY AUTOINCREMENT,
    department_name TEXT NOT NULL,
    budget REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS employees (
    employee_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    department_id INTEGER,
    salary REAL NOT NULL,
    hire_date TEXT NOT NULL,
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
);

-- Seed data for departments
INSERT INTO departments (department_name, budget) VALUES 
('Engineering', 500000.00),
('Data Science', 350000.00),
('Finance', 200000.00),
('Human Resources', 120000.00),
('Marketing', 180000.00);

-- Seed data for employees (50+ rows)
INSERT INTO employees (first_name, last_name, department_id, salary, hire_date) VALUES
('John', 'Doe', 1, 85000, '2022-01-15'), ('Jane', 'Smith', 1, 92000, '2021-03-22'),
('Michael', 'Johnson', 2, 95000, '2023-05-10'), ('Emily', 'Davis', 2, 105000, '2020-11-01'),
('David', 'Brown', 3, 75000, '2021-06-18'), ('Sarah', 'Miller', 4, 65000, '2022-08-30'),
('James', 'Wilson', 5, 70000, '2023-02-14'), ('Anna', 'Moore', 1, 88000, '2022-04-05'),
('Robert', 'Taylor', 2, 98000, '2021-09-12'), ('Linda', 'Anderson', 3, 78000, '2020-05-20'),
('William', 'Thomas', 4, 62000, '2023-07-11'), ('Elizabeth', 'Jackson', 5, 72000, '2022-10-01'),
('Richard', 'White', 1, 83000, '2021-12-05'), ('Barbara', 'Harris', 2, 110000, '2019-04-15'),
('Joseph', 'Martin', 3, 74000, '2022-02-28'), ('Susan', 'Thompson', 4, 67000, '2021-07-19'),
('Thomas', 'Garcia', 5, 69000, '2023-01-25'), ('Jessica', 'Martinez', 1, 91000, '2020-08-14'),
('Christopher', 'Robinson', 2, 102000, '2022-06-01'), ('Karen', 'Clark', 3, 76000, '2023-04-18'),
('Daniel', 'Rodriguez', 4, 64000, '2022-11-10'), ('Nancy', 'Lewis', 5, 71000, '2021-05-05'),
('Matthew', 'Lee', 1, 86000, '2023-03-20'), ('Lisa', 'Walker', 2, 97000, '2022-09-15'),
('Mark', 'Hall', 3, 79000, '2020-02-10'), ('Betty', 'Allen', 4, 66000, '2021-01-22'),
('Donald', 'Young', 5, 73000, '2023-08-05'), ('Sandra', 'King', 1, 89000, '2021-10-18'),
('George', 'Wright', 2, 115000, '2018-07-01'), ('Ashley', 'Lopez', 3, 77000, '2022-05-12'),
('Kenneth', 'Hill', 4, 63000, '2023-09-01'), ('Donna', 'Scott', 5, 68000, '2022-03-14'),
('Steven', 'Green', 1, 84000, '2020-06-30'), ('Carol', 'Adams', 2, 101000, '2021-11-25'),
('Edward', 'Baker', 3, 81000, '2019-12-15'), ('Michelle', 'Gonzalez', 4, 68000, '2022-01-10'),
('Brian', 'Nelson', 5, 74000, '2023-06-12'), ('Dorothy', 'Carter', 1, 93000, '2021-02-28'),
('Ronald', 'Mitchell', 2, 104000, '2020-09-05'), ('Carolyn', 'Perez', 3, 82000, '2022-07-22'),
('Anthony', 'Roberts', 4, 61000, '2023-10-15'), ('Amanda', 'Turner', 5, 75000, '2021-04-11'),
('Kevin', 'Phillips', 1, 87000, '2022-05-25'), ('Melissa', 'Campbell', 2, 96000, '2023-02-01'),
('Jason', 'Parker', 3, 80000, '2020-08-19'), ('Deborah', 'Evans', 4, 69000, '2021-03-15'),
('Jeff', 'Edwards', 5, 76000, '2022-12-01'), ('Amy', 'Collins', 1, 94000, '2019-11-10'),
('Ryan', 'Stewart', 2, 108000, '2021-06-05'), ('Nicole', 'Morris', 3, 83000, '2023-01-20');