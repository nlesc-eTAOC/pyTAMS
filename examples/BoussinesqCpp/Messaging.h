#include <iostream>
#include <fstream>
#include <stdio.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/**
 * @file Messaging.h
 * @author Lucas Esclapez
 * @date 2025-06-13
 * @brief Messaging classes and functions for IPC
 * @details
 *  An IPC between the python side implementing the TAMS API
 *  and the C++ side implementing the Boussinesq model.
 *  A limited number of message types are defined, to control
 *  the inputs and flow of the C++ model.
 * 
 */

/**
 * @brief Define message types (actions)
 */
enum MessageType {
  Null,               // Default, do nothing message
  SetWorkDir,         // Set the model workdir
  SetState,           // Pass a state file from Python to C++
  GetState,           // Pass a state file from C++ to python
  GetScore,           // Get the model score
  OneStep,            // Do a time step, without stochastic forcing
  OneStochStep,       // Pass a vector of stochastic forcing and do a time step
  SaveState,          // Trigger state saving
  Done,               // Signal to Python that the requested action is done
  Exit,               // Signal to exit C++ while loop, terminating the model
};

/**
 * @brief Message struct
 * @details
 *   A message is a struct containing an action type, a size of the message,
 *   and a pointer to the message data. 
 *   The data is a shared pointer to a char array, which is allocated on the heap.
 *   This struct is used to pass messages between the C++ and Python sides.
 */
struct Message {
  MessageType type{MessageType::Null};
  int size{0};
  std::unique_ptr<char[]> data;

  Message(MessageType t, int s) : type{t}, size{s} {
    if (s > 0) {
      data = std::make_unique<char[]>(s);
    }
  }

  Message(Message & msg) : type{msg.type}, size{msg.size} {
    if (msg.size > 0) {
      data = std::move(msg.data);
    }
  }

  ~Message() {
    data.reset();
  }
};

// Some default messages
Message exit_msg{MessageType::Exit, 0};
Message done_msg{MessageType::Done, 0};

/**
 * @brief TwoWayPipe class
 * @details
 *   A class to handle the IPC between the C++ and Python sides.
 *   The class creates two named pipes, one for reading and one for writing.
 *   The class uses the open and close system calls to handle the pipes.
 */
struct TwoWayPipe {
  /**
   * @brief Constructor
   * @details
   *   The constructor creates the named pipes and opens them for reading and writing.
   */
  TwoWayPipe(std::string id) : m_id{id},
    fifo_rd_str{"./ptoc_" + m_id},
    fifo_wr_str{"./ctop_" + m_id}
  {
    fifo_rd = (char *)fifo_rd_str.c_str();
    fifo_wr = (char *)fifo_wr_str.c_str();
    mkfifo(fifo_rd, 0777);
    mkfifo(fifo_wr, 0777);

    // Open the write pipe first, unlocking the python side 
    fwr = open(fifo_wr, O_WRONLY);
    // Then open the read pipe
    frd = open(fifo_rd, O_RDONLY);
  }

  /**
   * @brief Destructor
   * @details
   *   The destructor closes the named pipes.
   */
  ~TwoWayPipe() {
    close(fwr);
    close(frd);
  }

  /**
   * @brief get_message
   * @details
   *   The get_message function reads a message from the read pipe.
   *   It returns a message struct containing the message type, size, and data.
   */
  Message get_message() {
    int mtype{0};
    read(frd, &mtype, sizeof(int));

    int msize{0};
    read(frd, &msize, sizeof(int));

    Message msg((MessageType)mtype, msize);

    if (msize > 0) {
      int bytes_read = read(frd, msg.data.get(), msize);
    }

    return msg;
  }

  /**
   * @brief post_message
   * @details
   *   The post_message function writes a message to the write pipe.
   *   It takes a message struct containing the message type, size, and data.
   */
  void post_message(const Message &msg) {
    write(fwr, &msg.type, sizeof(int));
    write(fwr, &msg.size, sizeof(int));
    if (msg.size > 0) {
      write(fwr, msg.data.get(), msg.size);
    }
  }

  std::string m_id;
  std::string fifo_rd_str;
  std::string fifo_wr_str;
  char * fifo_rd; 
  char * fifo_wr;
  int fwr{0};
  int frd{0};
};
