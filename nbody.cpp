#include <chrono>
#include <cstdlib>
#include <iostream>

#include <sycl/sycl.hpp>

void dump_device_info(sycl::device device) {
  std::cout << "Running on device: "
            << device.get_info<sycl::info::device::name>() << "\n";
  /*
std::cout << "Vendor: " << device.get_info<sycl::info::device::vendor>()
  << "\n";
std::cout << "Driver Version: "
  << device.get_info<sycl::info::device::driver_version>() << "\n";
std::cout << "Max Compute Units: "
  << device.get_info<sycl::info::device::max_compute_units>() << "\n";
std::cout << "Global Memory Size: "
  << device.get_info<sycl::info::device::global_mem_size>() /
         (1024 * 1024)
  << " MB\n";
std::cout << "Device type: ";
switch (device.get_info<sycl::info::device::device_type>()) {
case sycl::info::device_type::cpu:
std::cout << "CPU\n";
break;
case sycl::info::device_type::gpu:
std::cout << "GPU\n";
break;
case sycl::info::device_type::accelerator:
std::cout << "Accelerator\n";
break;
default:
std::cout << "Unknown\n";
break;
}
*/
}

struct coordinates {
  double x;
  double y;
  double z;
};

struct velocity {
  double x;
  double y;
  double z;
};

// we need 3 of these y'know
struct body {
  double mass;
  coordinates position;
  velocity speed;

  body() {
    position.x = rand() % 16384;
    position.y = rand() % 16384;
    position.z = rand() % 16384;
    mass = rand() + 1;
    speed.x = 0;
    speed.y = 0;
    speed.z = 0;
  };
};

const double G = 6.674e-11;

const auto T = 10;

// number of bodies
constexpr int SIZE = 10000;

void updateBodies(std::vector<body>& bodies) {
  const auto body_count = bodies.size();

  // array for the partial sum of forces acting on the bodies
  std::vector<double> fx_sum(body_count, 0.0);
  std::vector<double> fy_sum(body_count, 0.0);
  std::vector<double> fz_sum(body_count, 0.0);

  // work on body pairs
  for (int i = 0; i < body_count; ++i) {
    for (int j = i + 1; j < body_count; ++j) {
      // get the difference between the axis positions, no need to take an
      // absolute value since we will square them
      const auto dx = bodies[i].position.x - bodies[j].position.x;
      const auto dy = bodies[i].position.y - bodies[j].position.y;
      const auto dz = bodies[i].position.z - bodies[j].position.z;
      const auto distSqr = std::pow(dx, 2) + std::pow(dy, 2) + std::pow(dz, 2);
      const auto distance = std::sqrt(distSqr);

      // get the force acting between the 2 bodies (mass times mass over
      // distance squared)
      const auto force = G * bodies[i].mass * bodies[j].mass / distSqr;

      // split the total force acting on the bodies onto the 3 components
      const auto fx = force * dx / distance;
      const auto fy = force * dy / distance;
      const auto fz = force * dz / distance;
      fx_sum[i] += fx;
      fy_sum[i] += fy;
      fz_sum[i] += fz;
      fx_sum[j] -= fx;
      fy_sum[j] -= fy;
      fz_sum[j] -= fz;
    }
  }

  // once we have all the forces we can update the velocity, first get the
  // accelleration f = ma -> a = f/m

  for (int i = 0; i < body_count; ++i) {
    auto* body = &bodies[i];
    const auto body_mass = body->mass;
    const auto ax = fx_sum[i] / body_mass;
    const auto ay = fy_sum[i] / body_mass;
    const auto az = fz_sum[i] / body_mass;

    // V = V0 + aT
    body->speed.x += ax * T;
    body->speed.y += ay * T;
    body->speed.z += az * T;

    // update the position, S = S0 + VT
    body->position.x += body->speed.x * T;
    body->position.y += body->speed.y * T;
    body->position.z += body->speed.z * T;
  }
}

void updateBodiesParallel(std::vector<body>& bodies) {
  try {
    sycl::queue cpuQueue(sycl::gpu_selector_v);

    // Create buffer from current host data each iteration
    sycl::buffer buffer(bodies.data(), sycl::range<1>(SIZE));
    // create buffers for force accumulation
    sycl::buffer<double> fx_buf(SIZE), fy_buf(SIZE), fz_buf(SIZE);

    for (int step = 0; step < 50; ++step) {
      cpuQueue.submit([&](sycl::handler& cgh) {
        sycl::accessor fx(fx_buf, cgh, sycl::write_only);
        sycl::accessor fy(fy_buf, cgh, sycl::write_only);
        sycl::accessor fz(fz_buf, cgh, sycl::write_only);

        cgh.parallel_for(sycl::range<1>(SIZE), [=](sycl::id<1> i) {
          fx[i] = 0.0;
          fy[i] = 0.0;
          fz[i] = 0.0;
        });
      });

      cpuQueue.wait();

      cpuQueue.submit([&](sycl::handler& cgh) {
        sycl::accessor bodiesAcc(buffer, cgh, sycl::read_write);
        sycl::accessor fx(fx_buf, cgh, sycl::read_write);
        sycl::accessor fy(fy_buf, cgh, sycl::read_write);
        sycl::accessor fz(fz_buf, cgh, sycl::read_write);

        cgh.parallel_for(sycl::range<1>(SIZE), [=](sycl::id<1> i) {
          for (int j = i + 1; j < SIZE; ++j) {
            double dx = bodiesAcc[i].position.x - bodiesAcc[j].position.x;
            double dy = bodiesAcc[i].position.y - bodiesAcc[j].position.y;
            double dz = bodiesAcc[i].position.z - bodiesAcc[j].position.z;
            const auto distSqr =
                (std::pow(dx, 2) + std::pow(dy, 2) + std::pow(dz, 2));
            const auto distance = sycl::sqrt(distSqr);

            // get the force acting between the 2 bodies (mass times mass over
            // distance squared)
            const auto force =
                G * bodiesAcc[i].mass * bodiesAcc[j].mass / distSqr;

            using atomic_t =
                sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>;

            // split the total force acting on the bodies onto the 3 components
            atomic_t(fx[i]) += force * dx / distance;
            atomic_t(fy[i]) += force * dy / distance;
            atomic_t(fz[i]) += force * dz / distance;
            // inverse forces
            atomic_t(fx[j]) -= force * dx / distance;
            atomic_t(fy[j]) -= force * dy / distance;
            atomic_t(fz[j]) -= force * dz / distance;
          }
        });
      });

      // Wait for kernel to finish
      cpuQueue.wait();

      cpuQueue.submit([&](sycl::handler& cgh) {
        sycl::accessor bodiesAcc(buffer, cgh, sycl::read_write);
        sycl::accessor fx(fx_buf, cgh, sycl::read_only);
        sycl::accessor fy(fy_buf, cgh, sycl::read_only);
        sycl::accessor fz(fz_buf, cgh, sycl::read_only);

        cgh.parallel_for(sycl::range<1>(SIZE), [=](sycl::id<1> i) {
          bodiesAcc[i].speed.x += (fx[i] / bodiesAcc[i].mass) * T;
          bodiesAcc[i].speed.y += (fy[i] / bodiesAcc[i].mass) * T;
          bodiesAcc[i].speed.z += (fz[i] / bodiesAcc[i].mass) * T;

          bodiesAcc[i].position.x += bodiesAcc[i].speed.x * T;
          bodiesAcc[i].position.y += bodiesAcc[i].speed.y * T;
          bodiesAcc[i].position.z += bodiesAcc[i].speed.z * T;
        });
      });

      // Wait for kernel to finish
      cpuQueue.wait();

      // Host accessor to sync data back and read updated positions
      sycl::host_accessor host_acc(buffer, sycl::read_only);

      std::cout << "Step " << step << ": Body at (" << host_acc[0].position.x
                << ", " << host_acc[0].position.y << ", "
                << host_acc[0].position.z << ")\n";
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
  }
}

int main() {
  std::vector<body> bodies;

  srand(0xDEADBEAF);

  for (int i = 0; i < SIZE; i++) {
    bodies.push_back(body());
  }

  auto bodycopy = bodies;

  auto start = std::chrono::high_resolution_clock::now();
  for (int step = 0; step < 50; ++step) {
    updateBodies(bodies);
    std::cout << "Step " << step << ": Body at (" << bodies[0].position.x
              << ", " << bodies[0].position.y << ", " << bodies[0].position.z
              << ")\n";
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = end - start;
  std::cout
      << "Time taken (scalar): "
      << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
      << " milliseconds\n";

  start = std::chrono::high_resolution_clock::now();

  updateBodiesParallel(bodycopy);

  end = std::chrono::high_resolution_clock::now();
  duration = end - start;
  std::cout
      << "Time taken (SYCL): "
      << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
      << " milliseconds\n";
}
