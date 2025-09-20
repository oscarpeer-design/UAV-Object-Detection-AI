using namespace std;
#include <iostream>
#include <cmath>
#include <vector>

struct Point {
    double x;
    double y;
    double z; //A value of -1 means empty space. In 3D space, this is behind the LIDAR camera.
    Point(): x(0), y(0), z(-1) {}//without constructor
    Point(double z) : x(0), y(0), z(z) {} //with only z passed
    Point(double x, double y, double z) : x(x), y(y), z(z) {} //with all passed
};

struct ScoredPoint { // Score for an optimal point, including both the clearance and trajectory components
    Point position;
    double clearanceScore;
    double trajectoryScore;
    double totalScore;
    ScoredPoint() : position(Point()), clearanceScore(0), trajectoryScore(0), totalScore(0) {}
    ScoredPoint(Point p, double c, double a, double t): position(p), clearanceScore(c), trajectoryScore(a), totalScore(t) {}
};

class Vector {
    private:
        double x;
        double y;
        double z;

    public:
        Vector() {
            x = 0;
            y = 0;
            z = 0;
        }

        Vector(double xComponent, double yComponent, double zComponent) {
            x = xComponent;
            y = yComponent;
            z = zComponent;
        }

        ~Vector() {}

        double average(double p1, double p2) {
            double avg = (p1 + p2) / 2;
            return avg;
        }

        Vector averageVector(Vector v) {
            //returns average inbetween u and v
            double xComp = average(xComponent(), v.xComponent());
            double yComp = average(yComponent(), v.yComponent());
            double zComp = average(zComponent(), v.zComponent());
            return Vector(xComp, yComp, zComp);
        }

        double vectorLength() {
            double length = 0;
            length = sqrt(x*x + y*y + z*z);
            return length;
        }

        Vector unitVector() {
            double length = vectorLength();
            double xComponent, yComponent, zComponent;
            //guard on division by zero
            if (length == 0) {
                return Vector(0, 0, 0);
            }
            xComponent = x / length;
            yComponent = y / length;
            zComponent = z / length;
            return Vector(xComponent, yComponent, zComponent);
        }

        Vector vectorOfTwoPoints(Point p1, Point p2) {
            //Where vector is FROM p1 TO p2
            double xComponent, yComponent, zComponent;
            xComponent = p2.x - p1.x;
            yComponent = p2.y - p1.y;
            zComponent = p2.z - p1.z;
            return Vector(xComponent, yComponent, zComponent);
        }

        double vectorDotProduct(Vector v) {
            double dotProduct = (v.x * x) + (v.y * y) + (v.z * z);
            return dotProduct;
        }

        double deviationScore(double cosTheta) {
            // cosTheta is is normalised between -1 and 1
            //-1 means 100%, 0 means 50% deviation, 1 means 0% deviation
            double deviationScore = (1.0 - cosTheta) / 2.0;
            return deviationScore;
        }

        double vectorDeviationScore(Vector v) {
            //Calculates % deviation of one vector from another
            //This is done by taking the dot product of the two vectors and dividing it by the length of both vectors
            // u.v = |u||v|cos(theta)
            //Therefore cos(theta) = u.v / (|u||v|)
            // This is normalised between -1 and 1
            //-1 means 100%, 0 means 50% deviation, 1 means 0% deviation
            double dotProduct = vectorDotProduct(v);
            double lengthU = vectorLength(); // this is the length of THIS ector
            double lengthV = v.vectorLength();
            double cosTheta = dotProduct / (lengthU * lengthV);
            cosTheta = max(-1.0, min(1.0, cosTheta)); // clamp between -1 and 1 to avoid rounding errors

            double score = deviationScore(cosTheta);
            return score;
        }

        double xComponent() {
            return x;
        }

        double yComponent() {
            return y;
        }

        double zComponent() {
            return z;
        }

};

class VideoFrameAnalysis {
    private:
        int maxRow = 10;
        int maxCol = 10;
        Point points[10][10];

    public:
        VideoFrameAnalysis() {

        }

        ~VideoFrameAnalysis() {}

        //This is stub
        void getTwoObjects() {
            double buildingFront = 5, buildingSide = 4, birdVal = 1;

            int row = 0;
            int col = 0;
            //initialising the building
            for (col = 1; col < 3; col++) {

                for (row = 0; row < maxRow; row++) {
                    points[row][col] = Point(buildingFront);
                }

            }
            col = 3;
            for (row = 0; row < maxRow; row++) {
                points[row][col] = Point(buildingSide);
            }
            //initialising the UAV in the frame
            points[1][4] = Point(birdVal);
            points[1][5] = Point(birdVal);
            points[1][6] = Point(birdVal);
            points[2][5] = Point(birdVal);
            points[2][6] = Point(birdVal);
            points[2][7] = Point(birdVal);

        }

        void printMatrixDepths() {
            double zCoord = 0;
            for (int row = 0; row < maxRow; row++) {
                for (int col = 0; col < maxCol; col++) {
                    zCoord = points[row][col].z;
                    cout << zCoord <<" ";
                }
                cout << "\n";
            }
        }

        void analyseFrame() {
            getTwoObjects(); //This is stub driver
            printMatrixDepths();
        }


        Point (*getPoints())[10] {
            return points;
        }

};

class ObjectAvoidance {
    private: 
        Vector trajectory;
        double trajectoryWeight = 0.6; //weight for staying on course
        double clearanceWeight = 0.4; //weight for staying closer to points with greater clearance
        double weightIncrement = 0.01;

        double alignmentExpectation = 0.7;
        double clearanceExpectation = 0.7;

    public:

        ObjectAvoidance() {
            trajectoryWeight = 0.6;
            clearanceWeight = getProbabilityComplement(trajectoryWeight);
            weightIncrement = 0.01;
            alignmentExpectation = 0.7;
            clearanceExpectation = 0.7;
        }

        ~ObjectAvoidance() {}
        
        double getProbabilityComplement(double p) {
            return 1.0 - p;
        }

        void smoothTrajectory(Vector newTrajectory) {
            double alpha = 0.3; // smoothing value
            trajectory = Vector(
                (1 - alpha) * trajectory.xComponent() + alpha * newTrajectory.xComponent(),
                (1 - alpha) * trajectory.yComponent() + alpha * newTrajectory.yComponent(),
                (1 - alpha) * trajectory.zComponent() + alpha * newTrajectory.zComponent()
            );
        }

        Vector getTrajectory() {
            return trajectory;
        }

        vector<vector<Point>> pointsWithoutObstacles(Point points[10][10]) {
            //Gets the ROWS that are empty (z-depth values of -1)
            int maxRow = 10;
            int maxCol = 10;

            vector<vector<Point>> emptyPointSet;

            for (int row = 0; row < maxRow; row++) {
                vector<Point> pointSet;   // stores consecutive empties in this row

                for (int col = 0; col < maxCol; col++) {
                    if (points[row][col].z == -1) {
                        // Still in an empty run → add point
                        pointSet.push_back(points[row][col]);
                    }
                    else {
                        // Hit an obstacle → if we collected empties, save them as an area
                        if (!pointSet.empty()) {
                            emptyPointSet.push_back(pointSet);
                            pointSet.clear();
                        }
                    }
                }

                // End of row → if there’s an unfinished empty run, save it
                if (!pointSet.empty()) {
                    emptyPointSet.push_back(pointSet);
                }
            }

            return emptyPointSet;
        }


        vector<double> deviationsFromTrajectory(Vector trajectory, Point currentPosition, vector<vector<Point>> emptyPointSet) {
            vector<double> deviationScores;
            Vector CP; //vector from current position to point
            //For each point, calculate the vector from the starting point and determine its deviation from the trajectory vector
            for (const auto& pointsArea: emptyPointSet) {
                for (const auto& point : pointsArea) {
                    CP = trajectory.vectorOfTwoPoints(currentPosition, point);
                    deviationScores.push_back(trajectory.vectorDeviationScore(CP));
                }
            }
            return deviationScores;
        }

        double distanceTwoPoints(Point candidatePoint, Point p) {
            double deltaX = candidatePoint.x - p.x;
            double deltaY = candidatePoint.y - p.y;
            double deltaZ = candidatePoint.z - p.z;

            double dist = sqrt(deltaX*deltaX + deltaY*deltaY + deltaZ*deltaZ);
            return dist;
        }

        double alignmentScore(double deviationScore) {
            // Invert deviation to alignment
            //This is decimal from 0 to 1.0
            return 1.0 - deviationScore;
        }

        double scorePoint(Point point, double alignmentScore, double clearanceScore) {
            double weightedScore = trajectoryWeight * alignmentScore + clearanceWeight * clearanceScore;
            return weightedScore;
        }

        double clampWeight(double weight) {
            //This clamps weights between 0 and 1
            if (weight < 0) return 0.0;
            if (weight > 1) return 1.0;
            return weight;
        }

        double applyIncrement(double weight, bool success) {
            if (success) {
                weight += weightIncrement;
            }
            else {
                weight -= weightIncrement;
            }
            return clampWeight(weight);
        }

        void normaliseWeights() {
            //Normalise between zero and 1
            double total = trajectoryWeight + clearanceWeight;
            if (total > 0.0) {
                trajectoryWeight /= total;
                clearanceWeight /= total;
            }
            else {
                // fallback: 60/40 split if both collapsed to 0
                trajectoryWeight = 0.6;
                clearanceWeight = 0.4;
            }
        }

        void updateWeights(bool successClearance, bool successTraj) {
            // Update trajectory based on trajectory success
            trajectoryWeight = applyIncrement(trajectoryWeight, successTraj);
            //Update clearance based on clearance success
            clearanceWeight = applyIncrement(clearanceWeight, successClearance);
            //Normalise so they sum to 1
            normaliseWeights();
        }

        ScoredPoint newOptimalPosition(vector<vector<Point>> emptyPointSet, vector<double> deviationScores) {
            //This balances decision making between the most empty spaces and lowest deviations from trajectory
            Point optimalPoint;
            double optimalScore = 0;
            double optimalAlignment = 0;
            double optimalClearance = 0;

            size_t pointSetIdx = 0;
            size_t pointIdx = 0;
            size_t deviationIdx = 0;
            size_t clearance = 0;
            size_t size = emptyPointSet.size();
            Point point;

            while (pointSetIdx <  size && deviationIdx < deviationScores.size()) {
                vector<Point> pointsArea = emptyPointSet[pointSetIdx];
                clearance = pointsArea.size(); //score based on distance between candidate point and surrounding empty points
                //Clearance gives the points in an area
                pointIdx = 0;
                while (pointIdx < clearance) {
                    double score = 0;
                    double clearanceScore = (double)clearance / (double)size;
                    double alignment = alignmentScore(deviationScores[deviationIdx]);
                    point = pointsArea[pointIdx];

                    score = scorePoint(point, alignment, clearance);
                    if (score > optimalScore) {
                        optimalScore = score;
                        optimalAlignment = alignment;
                        optimalClearance = clearanceScore;
                        optimalPoint = point;
                    }
                    pointIdx++;
                }
                deviationIdx++;

            }
            ScoredPoint optimisedScoredPoint(optimalPoint, optimalClearance, optimalAlignment, optimalScore);
            return optimisedScoredPoint;
        }
        
        bool trajectoryFollowed(double trajectoryScore) {
            return trajectoryScore >= alignmentExpectation;
        }

        bool adequateClearance(double clearanceScore) {
            return clearanceScore >= clearanceExpectation;
        }

        void alterHyperparameters(ScoredPoint optimalScoredPosition) {
            //This first determines success/failure, and then alters the weights
            bool successTraj = trajectoryFollowed(optimalScoredPosition.trajectoryScore);
            bool successClear = adequateClearance(optimalScoredPosition.clearanceScore);
            updateWeights(successClear, successTraj);
        }

        void avoidObstacles(Point points[10][10], Point currentPosition) {
            /*Steps in AI obstacle analysis algorithm:
            1. get all points that are empty space
            2. for each empty point, calculate deviation from trajectory
            3. select the point that has the most empty space and deviates least from the trajectory
            4.change the hyperparameters, including the trajectory
            */
            vector<vector<Point>> emptyPoints = pointsWithoutObstacles(points);
            vector<double> deviationScores = deviationsFromTrajectory(trajectory, currentPosition, emptyPoints);
            ScoredPoint optimisedScoredPosition = newOptimalPosition(emptyPoints, deviationScores);
            Vector alteredTrajectory = trajectory.vectorOfTwoPoints(currentPosition, optimisedScoredPosition.position);
            
            smoothTrajectory(alteredTrajectory); //altering trajectory values
            //altering decision-making parameters
            alterHyperparameters(optimisedScoredPosition);
        }

};

class EvasionSystem {
    public:
        EvasionSystem() {}

        ~EvasionSystem() {}

        void evadeObstacles() {
            cout << "Detecting\n";
            VideoFrameAnalysis detector;
            detector.analyseFrame();
            Point(*points)[10] = detector.getPoints();
            ObjectAvoidance avoider;
            Vector trajectory(0, 0, 10); //current trajectory in m/s
            Point currentPosition(10, 10, 10); //current position in m
            avoider.smoothTrajectory(trajectory);
            avoider.avoidObstacles(points, currentPosition);
            trajectory = avoider.getTrajectory();
        }

};

int main() {
    EvasionSystem evasionSystem;
    evasionSystem.evadeObstacles();

    return 0;
}
