#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

struct CalibOptions {
    cv::Size board_size{9, 6};
    float square_size = 0.025f;  // meter

    int max_iterations = 10;
    double target_rms = 0.6;
    double corner_error_threshold = 1.5;
    double image_error_threshold = 1.2;
    double min_valid_corner_ratio = 0.75;

    cv::TermCriteria subpix_criteria{cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 1e-3};
    cv::TermCriteria calib_criteria{cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 1e-8};
};

struct StereoImagePair {
    std::string left_path;
    std::string right_path;
};

struct StereoDataset {
    cv::Size image_size;
    std::vector<std::vector<cv::Point3f>> object_points;  // per image
    std::vector<std::vector<cv::Point2f>> left_points;
    std::vector<std::vector<cv::Point2f>> right_points;
    std::vector<int> original_indices;  // map to source list
};

struct CalibrationResult {
    double rms = std::numeric_limits<double>::infinity();
    cv::Mat K1, D1, K2, D2, R, T;
};

std::vector<cv::Point3f> BuildBoardCorners(const cv::Size& board_size, float square_size) {
    std::vector<cv::Point3f> pts;
    pts.reserve(board_size.area());
    for (int r = 0; r < board_size.height; ++r) {
        for (int c = 0; c < board_size.width; ++c) {
            pts.emplace_back(static_cast<float>(c) * square_size, static_cast<float>(r) * square_size, 0.0f);
        }
    }
    return pts;
}

bool LoadPairList(const std::string& list_file, std::vector<StereoImagePair>* pairs) {
    std::ifstream ifs(list_file);
    if (!ifs.is_open()) {
        std::cerr << "Failed to open pair list file: " << list_file << "\n";
        return false;
    }

    pairs->clear();
    std::string left, right;
    while (ifs >> left >> right) {
        pairs->push_back({left, right});
    }
    return !pairs->empty();
}

bool DetectCornersOneImage(const cv::Mat& gray,
                           const cv::Size& board_size,
                           const cv::TermCriteria& subpix_criteria,
                           std::vector<cv::Point2f>* corners) {
    corners->clear();
    const int flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
    const bool ok = cv::findChessboardCorners(gray, board_size, *corners, flags);
    if (!ok) {
        return false;
    }

    cv::cornerSubPix(gray, *corners, cv::Size(7, 7), cv::Size(-1, -1), subpix_criteria);
    return true;
}

StereoDataset BuildDataset(const std::vector<StereoImagePair>& pairs, const CalibOptions& opts) {
    StereoDataset dataset;
    const auto board_3d = BuildBoardCorners(opts.board_size, opts.square_size);

    for (size_t i = 0; i < pairs.size(); ++i) {
        cv::Mat left = cv::imread(pairs[i].left_path, cv::IMREAD_GRAYSCALE);
        cv::Mat right = cv::imread(pairs[i].right_path, cv::IMREAD_GRAYSCALE);
        if (left.empty() || right.empty()) {
            std::cerr << "Skip pair " << i << ": image read failed\n";
            continue;
        }

        if (dataset.image_size.empty()) {
            dataset.image_size = left.size();
        }
        if (left.size() != dataset.image_size || right.size() != dataset.image_size) {
            std::cerr << "Skip pair " << i << ": inconsistent image size\n";
            continue;
        }

        std::vector<cv::Point2f> c1, c2;
        const bool ok1 = DetectCornersOneImage(left, opts.board_size, opts.subpix_criteria, &c1);
        const bool ok2 = DetectCornersOneImage(right, opts.board_size, opts.subpix_criteria, &c2);
        if (!(ok1 && ok2)) {
            std::cerr << "Skip pair " << i << ": chessboard not found in both views\n";
            continue;
        }

        dataset.object_points.push_back(board_3d);
        dataset.left_points.push_back(c1);
        dataset.right_points.push_back(c2);
        dataset.original_indices.push_back(static_cast<int>(i));
    }
    return dataset;
}

CalibrationResult CalibrateStereoFisheye(const StereoDataset& data, const CalibOptions& opts) {
    CalibrationResult result;
    result.K1 = cv::Mat::eye(3, 3, CV_64F);
    result.K2 = cv::Mat::eye(3, 3, CV_64F);
    result.D1 = cv::Mat::zeros(4, 1, CV_64F);
    result.D2 = cv::Mat::zeros(4, 1, CV_64F);

    const int flags = cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC | cv::fisheye::CALIB_CHECK_COND;

    result.rms = cv::fisheye::stereoCalibrate(data.object_points,
                                              data.left_points,
                                              data.right_points,
                                              result.K1,
                                              result.D1,
                                              result.K2,
                                              result.D2,
                                              data.image_size,
                                              result.R,
                                              result.T,
                                              flags,
                                              opts.calib_criteria);
    return result;
}

struct ImageErrorStats {
    std::vector<double> mean_corner_error;    // per image
    std::vector<double> max_corner_error;     // per image
    std::vector<std::vector<char>> inlier_mask;  // 1=inlier corner
};

ImageErrorStats ComputeReprojectionErrors(const StereoDataset& data,
                                          const CalibrationResult& calib,
                                          double corner_error_threshold) {
    ImageErrorStats stats;
    const size_t n = data.object_points.size();
    stats.mean_corner_error.resize(n, 0.0);
    stats.max_corner_error.resize(n, 0.0);
    stats.inlier_mask.resize(n);

    for (size_t i = 0; i < n; ++i) {
        cv::Mat rvec, tvec;
        cv::fisheye::solvePnP(data.object_points[i],
                              data.left_points[i],
                              calib.K1,
                              calib.D1,
                              rvec,
                              tvec,
                              false,
                              cv::SOLVEPNP_ITERATIVE);

        std::vector<cv::Point2f> proj_left, proj_right;
        cv::fisheye::projectPoints(data.object_points[i], proj_left, rvec, tvec, calib.K1, calib.D1);

        cv::Mat Rl;
        cv::Rodrigues(rvec, Rl);
        cv::Mat Rr = calib.R * Rl;
        cv::Mat tr = calib.R * tvec + calib.T;
        cv::Mat rvec_r;
        cv::Rodrigues(Rr, rvec_r);
        cv::fisheye::projectPoints(data.object_points[i], proj_right, rvec_r, tr, calib.K2, calib.D2);

        double err_sum = 0.0;
        double err_max = 0.0;
        stats.inlier_mask[i].assign(data.object_points[i].size(), 1);

        for (size_t j = 0; j < data.object_points[i].size(); ++j) {
            const double el = cv::norm(data.left_points[i][j] - proj_left[j]);
            const double er = cv::norm(data.right_points[i][j] - proj_right[j]);
            const double e = 0.5 * (el + er);

            err_sum += e;
            err_max = std::max(err_max, e);
            if (e > corner_error_threshold) {
                stats.inlier_mask[i][j] = 0;
            }
        }

        stats.mean_corner_error[i] = err_sum / static_cast<double>(data.object_points[i].size());
        stats.max_corner_error[i] = err_max;
    }
    return stats;
}

StereoDataset FilterDataset(const StereoDataset& data,
                            const ImageErrorStats& stats,
                            const CalibOptions& opts,
                            bool* removed_anything) {
    StereoDataset out;
    out.image_size = data.image_size;
    *removed_anything = false;

    for (size_t i = 0; i < data.object_points.size(); ++i) {
        std::vector<cv::Point3f> obj;
        std::vector<cv::Point2f> lft, rgt;
        obj.reserve(data.object_points[i].size());
        lft.reserve(data.object_points[i].size());
        rgt.reserve(data.object_points[i].size());

        for (size_t j = 0; j < data.object_points[i].size(); ++j) {
            if (stats.inlier_mask[i][j]) {
                obj.push_back(data.object_points[i][j]);
                lft.push_back(data.left_points[i][j]);
                rgt.push_back(data.right_points[i][j]);
            } else {
                *removed_anything = true;
            }
        }

        const double valid_ratio = static_cast<double>(obj.size()) / static_cast<double>(data.object_points[i].size());
        const bool drop_image = (valid_ratio < opts.min_valid_corner_ratio) ||
                                (stats.mean_corner_error[i] > opts.image_error_threshold);
        if (drop_image) {
            std::cout << "Drop image pair idx=" << data.original_indices[i]
                      << " valid_ratio=" << std::fixed << std::setprecision(3) << valid_ratio
                      << " mean_error=" << stats.mean_corner_error[i] << "\n";
            *removed_anything = true;
            continue;
        }

        out.object_points.push_back(std::move(obj));
        out.left_points.push_back(std::move(lft));
        out.right_points.push_back(std::move(rgt));
        out.original_indices.push_back(data.original_indices[i]);
    }

    return out;
}

std::optional<CalibrationResult> RobustStereoCalibrate(StereoDataset dataset, const CalibOptions& opts) {
    if (dataset.object_points.size() < 5) {
        std::cerr << "Not enough valid pairs after detection.\n";
        return std::nullopt;
    }

    CalibrationResult best;
    for (int iter = 0; iter < opts.max_iterations; ++iter) {
        if (dataset.object_points.size() < 5) {
            std::cerr << "Stopped: not enough pairs to continue at iter " << iter << "\n";
            break;
        }

        CalibrationResult calib = CalibrateStereoFisheye(dataset, opts);
        ImageErrorStats stats = ComputeReprojectionErrors(dataset, calib, opts.corner_error_threshold);

        const double mean_img_err = std::accumulate(stats.mean_corner_error.begin(), stats.mean_corner_error.end(), 0.0) /
                                    static_cast<double>(stats.mean_corner_error.size());

        std::cout << "Iter " << iter << " rms=" << calib.rms << " mean_img_err=" << mean_img_err
                  << " remain_pairs=" << dataset.object_points.size() << "\n";

        best = calib;
        if (calib.rms <= opts.target_rms && mean_img_err <= opts.image_error_threshold) {
            std::cout << "Converged: reprojection error is in acceptable range.\n";
            return best;
        }

        bool removed_anything = false;
        StereoDataset filtered = FilterDataset(dataset, stats, opts, &removed_anything);
        if (!removed_anything) {
            std::cout << "No more outliers can be removed; stop iteration.\n";
            return best;
        }
        dataset = std::move(filtered);
    }
    return best;
}

void PrintUsage() {
    std::cout << "Usage:\n"
              << "  ./fisheye_stereo_calib pair_list.txt board_w board_h square_size\n\n"
              << "pair_list.txt format: each line contains <left_image_path> <right_image_path>\n";
}

int main(int argc, char** argv) {
    if (argc < 5) {
        PrintUsage();
        return 1;
    }

    CalibOptions opts;
    opts.board_size = cv::Size(std::stoi(argv[2]), std::stoi(argv[3]));
    opts.square_size = std::stof(argv[4]);

    std::vector<StereoImagePair> pairs;
    if (!LoadPairList(argv[1], &pairs)) {
        return 2;
    }

    StereoDataset data = BuildDataset(pairs, opts);
    std::cout << "Detected valid stereo pairs: " << data.object_points.size() << " / " << pairs.size() << "\n";

    auto calib = RobustStereoCalibrate(std::move(data), opts);
    if (!calib.has_value()) {
        return 3;
    }

    std::cout << "\n=== Final stereo extrinsics ===\n";
    std::cout << "R:\n" << calib->R << "\n";
    std::cout << "T:\n" << calib->T << "\n";
    std::cout << "RMS: " << calib->rms << "\n";

    return 0;
}
