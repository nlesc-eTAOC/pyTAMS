#include "Boussinesq.h"
#include "Messaging.h"

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

struct CommandLineArgs {
    std::unordered_map<std::string, std::string> args;

    CommandLineArgs(int argc, char* argv[]) {
        parse(argc, argv);
    }

    void parse(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg.rfind("--", 0) == 0 && i + 1 < argc) {
                std::string key = arg.substr(2);
                std::string value = argv[i + 1];
                if (value.rfind("--", 0) != 0) { // not another flag
                    args[key] = value;
                    ++i; // skip value
                } else {
                    args[key] = "true"; // flag with no value
                }
            } else if (arg.rfind("-", 0) == 0 && arg.size() == 2 && i + 1 < argc) {
                std::string key = arg.substr(1);
                std::string value = argv[i + 1];
                if (value.rfind("-", 0) != 0) {
                    args[key] = value;
                    ++i;
                } else {
                    args[key] = "true";
                }
            }
        }
    }

    std::string get(const std::string& key, const std::string& default_val = "") const {
        auto it = args.find(key);
        return it != args.end() ? it->second : default_val;
    }

    bool has(const std::string& key) const {
        return args.find(key) != args.end();
    }
};

int main(int argc, char * argv[]) {
    // Parse command line arguments   
    CommandLineArgs args(argc, argv);
    int M = atoi(args.get("M", "40").c_str());
    int N = atoi(args.get("N", "80").c_str());
    int K = atoi(args.get("K", "4").c_str());
    double dt = atof(args.get("dt", "0.005").c_str());
    double eps = atof(args.get("eps", "0.01").c_str());
    std::string pipe_id = args.get("pipe_id", "000000");

    Boussinesq2D Boussinesq(M, N, K, eps, dt);

    // Hard-coded ON and OFF state for computing the score function
    Boussinesq.init_score("./ONState.bin", "./OFFState.bin");
    TwoWayPipe pipe(pipe_id);

    int maxloop = 200000;
    int loop = 0;
    // While loop keeping the model running, waiting for
    // commands from the python code
    while (loop < maxloop) {
        auto action = pipe.get_message();

        // Break the loop when triggered
        if (action.type == MessageType::Exit) {
          pipe.post_message(exit_msg);
          break;
        }

        // Set the model workdir
        if (action.type == MessageType::SetWorkDir) {
          std::string workdir(action.data.get());
          Boussinesq.set_workdir(workdir);
        }

        // Trigger saving the model state
        if (action.type == MessageType::SaveState) {
          Boussinesq.m_saveState = 1;
        }

        // Pass a new state to the model
        if (action.type == MessageType::SetState) {
          std::string statefile(action.data.get());
          pipe.post_message(done_msg);
          Boussinesq.set_statefile(statefile);
        }

        // Return the state file to python
        if (action.type == MessageType::GetState) {
          std::string statefile = Boussinesq.m_statefile;
          Message donewithstate{MessageType::Done, static_cast<int>(statefile.size())};
          memcpy(donewithstate.data.get(), statefile.c_str(), statefile.size());
          pipe.post_message(donewithstate);
        }

        // Return the model score to python
        if (action.type == MessageType::GetScore) {
          double score = Boussinesq.get_score();
          Message donewithscore{MessageType::Done, sizeof(double)};
          memcpy(donewithscore.data.get(), &score, sizeof(double));
          pipe.post_message(donewithscore);
        }

        // Do a single step, without stochastic forcing 
        if (action.type == MessageType::OneStep) {
          double score = Boussinesq.one_step();
          Message donewithscore{MessageType::Done, sizeof(double)};
          memcpy(donewithscore.data.get(), &score, sizeof(double));
          pipe.post_message(donewithscore);
        }

        // Do a single step, with stochastic forcing
        if (action.type == MessageType::OneStochStep) {
          int vlen = action.size / sizeof(double);
          std::vector<double> noise_dbl(vlen, 0.0);
          noise_dbl.assign((double *)action.data.get(), (double *)action.data.get() + vlen);
          Eigen::VectorXd noise = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(noise_dbl.data(), vlen);
          double score = Boussinesq.one_step(noise);
          Message donewithscore{MessageType::Done, sizeof(double)};
          memcpy(donewithscore.data.get(), &score, sizeof(double));
          pipe.post_message(donewithscore);
        }

        loop++;
    }

    return 0;
}
