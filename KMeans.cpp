#include "KMeans.hpp"
#include <random>
#include <omp.h>
double KMeans::euclidean_distance(BasePoint::Point center, BasePoint::Point dot){
    return sqrt((center.x-dot.x)*(center.x-dot.x)+(center.y-dot.y)*(center.y-dot.y)+(center.z-dot.z)*(center.z-dot.z));
}
KMeans::KMeans::KMeans(double epsilon, int maxIterations, int clusters) {
    this->epsilon=epsilon;
    this->clusters=clusters;
    this->maxIterations=maxIterations;
    std::vector<BasePoint::Point> pt;
    for (int i = 0; i < clusters; ++i) {
        clusterData.push_back(pt);
    }
}

void KMeans::KMeans::getRandomCenter() {
    center.clear();
    //----partly random center---------------------------------------------------//
    long areaLower=0;
    long areaUpper=points.size()/clusters;
    long areaAdder=areaUpper;

    for (int i = 0; i < clusters; ++i) {
        std::uniform_int_distribution<long long> dist(areaLower, areaUpper);
        auto param=points[dist(rand_num)];
        param.center=i;
        center.push_back(param);
        areaLower+=areaAdder;
        areaUpper+=areaAdder;
    }
    //-------------------full random center--------------

    //    std::uniform_int_distribution<long long> dist(0, points.size());
    //    for (int i = 0; i < clusters; ++i) {
    //        auto param=points[dist(rand_num)];
    //        param.center=i;
    //        center.push_back(param);
    //    }
}

void KMeans::KMeans::createClusters() {
    int closedCenterID;
    double minDistance;
    double distance;
#pragma omp parallel for private(closedCenterID,minDistance,distance) shared(points) default(none)
    for (auto &point:points) {
        closedCenterID=0;
        minDistance= euclidean_distance(point, center[0]);
        for (int j = 0; j < clusters; ++j) {
            distance= euclidean_distance(point,center[j]);
            if (distance<minDistance){
                minDistance=distance;
                closedCenterID=j;
            }
        }
        point.center=closedCenterID;
    }
    for (const auto&point:points){
        clusterData[point.center].push_back(point);
    }
}
/**
 * @todo if in need to parallel?
 * @return
 */
std::vector<BasePoint::Point> KMeans::KMeans::updateCenter() {
    std::vector<BasePoint::Point> newCenter;
    BasePoint::Point paramPoint;
    for (int i = 0; i < clusterData.size(); ++i) {
        paramPoint.initToZero();
        for (int j = 0; j < clusterData[i].size(); ++j) {
            paramPoint.x+=clusterData[i][j].x;
            paramPoint.y+=clusterData[i][j].y;
            paramPoint.z+=clusterData[i][j].z;
            paramPoint.center=clusterData[i][j].center;
        }
        paramPoint.x=paramPoint.x/double(clusterData[i].size());
        paramPoint.y=paramPoint.y/double(clusterData[i].size());
        paramPoint.z=paramPoint.z/double(clusterData[i].size());
        newCenter.push_back(paramPoint);
    }
    return newCenter;
}
/**
 * @todo in need to parallel??
 * @param center
 * @param newCenters
 * @return
 */
bool KMeans::KMeans::hasCloseCenterBellowEpsilon(BasePoint::Point center, const std::vector<BasePoint::Point> &newCenters){
    bool belowEpsilon= false;
    for(const auto&newCenter:newCenters){
        if (euclidean_distance(center,newCenter)<=epsilon){
            belowEpsilon= true;
        }
    }
    return belowEpsilon;
}
/**
 * @todo in need to parallel??
 * @param newCenters
 * @return
 */
bool KMeans::KMeans::convergence(const std::vector<BasePoint::Point> &newCenters) {
    bool conv= true;
    for (int i = 0; i < center.size(); ++i) {
        if(!hasCloseCenterBellowEpsilon(center[i], newCenters)){
            conv= false;
        }
    }
    if (conv){
        std::cout<<"---------successfully converged---------------"<<std::endl;
    }
    return conv;
}
void KMeans::KMeans::setData(const std::vector<BasePoint::Point> &pointsSet) {
    this->points=pointsSet;
}
void KMeans::KMeans::KMeansRun() {
    double start,end,average,fStart,fStop;
    average=0.0;
    getRandomCenter();
    createClusters();
    fStart=omp_get_wtime();
    for (int i = 0; i < maxIterations; ++i) {
        start=omp_get_wtime();
        auto newCenters=updateCenter();
        if (convergence(newCenters)){
            break;
        }
        std::cout<<"Iteration: "<<i<<" th Center update"<<std::endl;
        center.clear();
        center=newCenters;
        for (auto &clusterEle:clusterData){
            clusterEle.clear();
        }
        createClusters();
        end=omp_get_wtime()-start;
        std::cout<<"Iteration time (s):"<<end<<std::endl;
        if (i==0){
            average=end;
        } else{
            average=(average+end)/2;
        }
    }
    fStop=omp_get_wtime()-fStart;
    std::cout<<"--------------------------------------------------"<<std::endl;
    std::cout<<"*  Iteration end average time (s): "<<average<<"  "<<std::endl;
    std::cout<<"*  Iteration total time (s): "<<fStop<<"          "<<std::endl;
    std::cout<<"--------------------------------------------------"<<std::endl;

}

const std::vector<std::vector<BasePoint::Point>> &KMeans::KMeans::getClusterData() {
    return clusterData;
}



