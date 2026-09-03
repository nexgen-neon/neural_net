import math

class StudentsNet:

    def __init__(self, learning_rate=0.01):
        
            self.learning_rate = learning_rate

            self.w1 = 0.10
            self.w2 = 0.20
            self.w3 = 0.20
            self.w4 = 0.10

            self.b1 = 0.10
            self.b2 = 0.10

            
            self.w5 = 0.10
            self.w6 = 0.20

            self.b3 = 0.10

            self.w7 = 0.15
            self.w8 = 0.25

            self.b4 = 0.10

    @staticmethod
    def sigmoid(value):

        if value < -700:
            return 0.0

        if value > 700:
            return 1.0

        return 1.0 / (1.0 + math.exp(-value))

    @staticmethod
    def sigmoid_derivative(value):
        return value * (1.0 - value)

    def forward(self, x1, x2):

            
            z1 = x1 * self.w1 + x2 * self.w2 + self.b1
            h1 = self.sigmoid(z1)

        
            z2 = x1 * self.w3 + x2 * self.w4 + self.b2
            h2 = self.sigmoid(z2)

        
            z3 = h1 * self.w5 + h2 * self.w6 + self.b3

            z4 = h1* self.w7 + h2 * self.w8 + self.b4

            result_output = self.sigmoid(z3)
            grade_output = self.sigmoid(z4)


            return {
                "z1": z1,
                "z2": z2,
                "z3": z3,
                "z4": z4,
                "h1": h1,
                "h2": h2,
                "result_output": result_output,
                "grade_output" : grade_output,
            }

    def predict(self, x1, x2):

            result = self.forward(x1, x2)

            return (
                result["result_output"],
                result["grade_output"]
            )
    
    def predict_result(self,x1,x2):
            result_output,_ = self.predict(x1,x2)

            if result_output > 0.5:
                return "passed"
            return "failed" 

    def grade_output(self,x1,x2):
            _,grade_output = self.predict(x1,x2)
            
            if grade_output >= 0.75:
                return "A"
            elif grade_output>= 0.25:
                return "B"
            else:
                return "C"

    def train(self, x1, x2, result_target, grade_target):

       
            result = self.forward(x1, x2)

            h1 = result["h1"]
            h2 = result["h2"]
            result_output = result["result_output"]
            grade_output = result["grade_output"]


            result_error = result_output - result_target
            grade_error = grade_output - grade_target

            result_loss = result_error ** 2
            grade_loss = grade_error ** 2

            d_result_loss = 2 * (result_output - result_target)

            sigmoid_result_output_derivative = self.sigmoid_derivative(
            result_output
            )

            dz3 = (
                d_result_loss * sigmoid_result_output_derivative
            )

            db3 = dz3

            dw5 = dz3 * h1
            dw6 = dz3 * h2

            d_grade_loss = 2* (grade_output - grade_target)
            sigmoid_grade_output_derivative = self.sigmoid_derivative(grade_output)
            dz4 = (d_grade_loss *sigmoid_grade_output_derivative)

            db4 = dz4 
            dw7 = dz4 * h1
            dw8 = dz4 * h2


            dh1 = dz3 * self.w5 + dz4 * self.w7
            dh2 = dz3 * self.w6 + dz4 * self.w8

            sigmoid_h1_derivative = self.sigmoid_derivative(h1)
            sigmoid_h2_derivative = self.sigmoid_derivative(h2)

            dz1 = dh1 * sigmoid_h1_derivative
            dz2 = dh2 * sigmoid_h2_derivative

            db1 = dz1
            db2 = dz2

            dw1 = dz1 * x1
            dw2 = dz1 * x2

            dw3 = dz2 * x1
            dw4 = dz2 * x2

        

            self.w1 -= self.learning_rate * dw1
            self.w2 -= self.learning_rate * dw2
            self.w3 -= self.learning_rate * dw3
            self.w4 -= self.learning_rate * dw4

            self.w5 -= self.learning_rate * dw5
            self.w6 -= self.learning_rate * dw6
            self.w7 -= self.learning_rate * dw7
            self.w8 -= self.learning_rate * dw8

            self.b1 -= self.learning_rate * db1
            self.b2 -= self.learning_rate * db2
            self.b3 -= self.learning_rate * db3
            self.b4 -= self.learning_rate * db4

            total_loss = ((result_loss + grade_loss)/2.0)

            return total_loss

    def classification(self, x1, x2):

        result_output, grade_output = self.predict(x1, x2)

        if result_output >= 0.5:
            result = "passed"
        else:
            result = "failed"

        if grade_output >= 0.75:
            grade = "A"
        elif grade_output >= 0.25:
            grade = "B"
        else:
            grade = "C"

        return (
        result_output,
        result,
        grade_output,
        grade
        )