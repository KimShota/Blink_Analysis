#include <cmath>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <boost/math/distributions/students_t.hpp>
using namespace std; 

//function to implement pearson correlation coefficient
double pearson_correlation_coefficient (const vector<double>& x, const &vector<double>& y){
    int n = x.size(); 
    if (n != (int)y.size() || n == 0){
        throw invalid_argument("Two vectors must have the same non-zero length"); 
    }

    //Calculate the mean values for x and y 
    double mean_x = accumulate(x.begin(), x.end(), 0.0) / n; 
    double mean_y = accumulate(y.begin(), y.end(), 0.0) / n;

    double numerator = 0.0; 
    double sum_x = 0.0; 
    double sum_y = 0.0; 

    
    for (int i = 0; i < n; i++){
        double diff_x = x[i] - x_mean; 
        double diff_y = y[i] - y_mean; 
        numerator += diff_x * diff_y; 
        sum_x += diff_x * diff_x; 
        sum_y += diff_y * diff_y;
    }

    double denominator = sqrt(sum_x * sum_y); 

    if (denominator == 0){
        throw runtime_error("Division by 0 is not allowed"); 
    }
    
    double r_value = numerator / denominator; 

    return r_value; 
}

//function to calculate p value for pearson 
double p_value_pearson(const vector<double>& x, const vector<double>& y){
    int n = x.size(); 
    double r_value = pearson_correlation_coefficient(x, y);
    
    int degrees_of_freedom = n - 2; 
    double t = r_value * sqrt(double(degrees_of_freedom) / (1 - r_value * r_value));

    boost::math::students_t dist(degrees_of_freedom); //create an object for student's t distribution 

    double p_value = 2 * (1 - cdf(dist, fabs(t)));
    
    return p_value; 
}

//function to rankify elements of vector 
vector<double> rankify(const vector<double>& data){
    int n = data.size(); 
    vector<pair<double, int>> temp(n); //create vector with each element being a container 
    
    //save the original index 
    for (int i = 0; i < n; i++){
        temp[i] = {data[i], i}; 
    }

    //sort the temporary data
    sort(temp.begin(), temp.end()); 

    vector<int> ranks(n); 

    //assign rank to each original index 
    for (int i = 0; i < n; i++){
        ranks[temp[i].second] = i + 1; 
    }

    return ranks; 
}

//function to implement spearman rank correlation coefficient 
double spearman_rank_correlation(const vector<double>& x, const vector<double>& y){
    int n = x.size(); 
    if (n != (int)y.size() || n == 0){
        throw invalid_argument("Two vectors must have the same non-zero length");
    }

    x_rank = rankify(x); 
    y_rank = rankify(y);

    double diff = 0.0; 
    double sum = 0.0; 
    
    for (int i = 0; i < n; i++){
        diff = x_rank - y_rank; 
        sum += diff * diff; 
    }

    double rho = 1.0 - ((6.0 * sum) / (n * (n*n -1))); 

    return rho; 
}

//function to calculate p value for spearman 
double p_value_spearman(const vector<double>& x, const vector<double>& y){
    int n = x.size(); 
    double rho = spearman_rank_correlation(x, y); 

    int df = n - 2; //degrees of freedom is 2 since we have two variables 
    double t = rho * sqrt(df / (1 - rho * rho));
    
    boost::math::students_t dist(df);
    
    double p_value = 2 * (1 - cdf(dist, fabs(t))); // calculate p value
    return p_value;  
}

//function to implement kendall tau
double kendall_tau (const vector<double>& x, const vector<double>& y){
    int n = x.size(); 
    if (n != y.size() || n == 0){
        throw invalid_argument("Two vectors must have the same non-zero length"); 
    }

    int concordant = 0; 
    int discordant = 0; 
    int tie_x = 0; 
    int tie_y = 0; 

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            double diff_x = x[i] - x[j]; 
            double diff_y = y[i] - y[j]; 

            if (diff_x == 0 && diff_y == 0){
                continue; 
            } else if (diff_x == 0){
                tie_x++; 
            } else if (diff_y == 0){
                tie_y++; 
            } else if (diff_x * diff_y > 0){
                concordant++; 
            } else if (diff_x * diff_y < 0){
                discordant++; 
            }
        }
    }

    double numerator = concordant - discordant; 
    double denominator = sqrt((concordant+discordant+tie_x)(concordant+discordant+tie_y));
    
    if (denominator == 0){
        throw runtime_error("Division by 0 is not allowed"); 
    }

    double tau = numerator / denominator; 

    return tau; 
}

double p_value_kendalltau(const vector<double>& x, const vector<double>& y){
    int n = x.size(); 
    double tau = kendall_tau(x, y);

    int df = n - 2; 

    double t = tau * sqrt(df / (1 - tau * tau));
    
    boost::math::students_t t_distribution(df);
    
    double p_value = 2 * (1 - cdf(t_distribution, fabs(t))); 
    return p_value; 
}

//function to implement distance correlation 
double distance_correlation(const vector<double>& x, const vector<double>& y){
    int n = x.size(); 
    if n != y.size() || n == O){
        throw invalid_argument("Two vectors must have the same non-zero length"); 
    }

    //create n*n matrix
    vector<vector<double>> diff_x(n, vector<double>(n));
    vector<vector<double>> diff_y(n, vector<double>(n)); 

    for (int x = 0; x < n; x++){
        for (int y = 0; y < n; y++){
            diff_x[i][j] = fabs(x[i] - x[j]); 
            diff_y[i][j] = fabs(y[i] - y[j]); 
        }
    }

    //double center the distance matrices
    auto double_center = [&] (vector<vector<double>>& dis_mat){
        vector<double> row_mean(n, 0.0), col_mean(n, 0.0);
        double grand_mean = 0.0;

        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                row_mean[i] += dis_mat[i][j]; 
                col_mean[j] += dis_mat[i][j]; 
                grand_mean += dis_mat[i][j]; 
            }
        }

        for (int i = 0; i < n; i++){
            row_mean[i] / n; 
        }
        for (int j = 0; j < n; j++){
            col_mean[j] / n;
        }
        grand_mean /= (n * n); 

        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                dis_mat[i][j] = dis_mat[i][j] - row_mean[i] - col_mean[j] + grand_mean; 
            }
        }
    };

    double_center(x); 
    double_center(y); 

    double dCov = 0.0, dVar_x = 0.0, dVar_y = 0.0; 

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            dCov += x[i][j] * y[i][j]; 
            dVar_x += x[i][j] * x[i][j]; 
            dVar_y += y[i][j] * y[i][j];
        }
    }

    dCov /= (n*n); 
    dVar_x /= (n*n); 
    dVar_y /= (n*n); 

    double dCor = dCov / sqrt(dVar_x * dVar_y); 

    return dCor;

}

//function to calculate p value for distance correlation 
double p_value_distance(const vector<double>& x, const vector<double>& y){
    int n = x.size(); 
    double dCor = distance_correlation(x, y);

    int df = n - 2; 

    double t = dCor * sqrt(df / (1 - dCor * dCor));
    
    boost::math::students_t t_distribution(df);
    
    double p_value = 2 * (1 - cdf(t_distribution, fabs(t))); 
    return p_value; 
}


int main(){
    vector<double> x, y; 


    return 0; 
}






