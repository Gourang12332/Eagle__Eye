"use client";

import Image from "next/image";

export default function Home() {
  const scrollToSection = (id: string) => {
    const section = document.getElementById(id);
    if (section) {
      section.scrollIntoView({ behavior: "smooth" });
    } else {
      console.error(`Element with id "${id}" not found.`);
    }
  };

  return (
    <div className="min-h-screen text-white">
      {/* Navbar */}
      <nav className="bg-gray-500 bg-opacity-10 left-0 flex justify-between items-center px-10 py-5 fixed w-full top-0 z-50 shadow-lg">
        <h1 className="text-3xl font-bold">EagleEyes</h1>
        <div className="flex gap-6">
          <button
            onClick={() => scrollToSection("home")}
            className="hover:text-black text-1xl mr-10 ml-0"
          >
            Home
          </button>
          <button
            onClick={() => scrollToSection("crowd")}
            className="hover:text-black text-1xl mr-10"
          >
            Crowd Management
          </button>
          <button
            onClick={() => scrollToSection("target")}
            className="hover:text-black text-1xl mr-10"
          >
            Target Detection
          </button>
          <button
            onClick={() => scrollToSection("crime")}
            className="hover:text-black text-1xl mr-10"
          >
            Crime Prevention
          </button>
          <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-xl border-blue-50">
            Register
          </button>
          <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg border-blue-50">
            Sign In
          </button>
        </div>
      </nav>

      <section
        id="home"
        className="flex items-center justify-center h-screen px-10 pt-20 bg-gray-100"
      >
        <div className="text-left text-gray-900">
          <h2 className="text-4xl font-bold mb-4">Advanced CCTV Monitoring</h2>
          <p className="text-lg mb-6">
            Real-time surveillance with AI-based anomaly detection <br />
            and high-definition monitoring for{" "}
            <b>
              crowd management, <br /> crime prevention,
            </b>{" "}
            and <b>work monitoring</b>
          </p>
          <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg">
            Learn More
          </button>
        </div>

        <div className="ml-10">
          <Image
            src="/cctv4.webp"
            alt="CCTV Monitoring"
            width={500}
            height={400}
            className="rounded-lg shadow-lg"
          />
        </div>
      </section>

      <section
        id="crowd"
        className="flex items-center justify-center h-screen px-10 bg-gray-100"
      >
        <Image
          src="/crowd3.jpg"
          alt="Crowd Management"
          width={500}
          height={400}
          className="rounded-lg shadow-lg ml-10"
        />
        <div className="text-left text-gray-900 ml-20">
          <h2 className="text-4xl font-bold mb-4">Crowd Management</h2>
          <p className="text-lg mb-6">
            AI-powered analytics to monitor the crowd at important places <br />
            and to prevent any condition of stampede.
          </p>
          <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg mt-5">
            Manage Crowd
          </button>
        </div>
      </section>

      <section
        id="target"
        className="flex items-center justify-center h-screen px-10 bg-gray-100"
      >
        <div className="text-left text-gray-900 ml-10 mr-10">
          <h2 className="text-4xl font-bold mb-4">Target Detection</h2>
          <p className="text-lg mb-6">
            Advanced person recognition and tracking to identify potential{" "}
            <br />
            criminal or missing person in large and small cities.
          </p>
          <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg">
            Detect Target
          </button>
        </div>

        <Image
          src="/rArea.png"
          alt="Target Detection"
          width={500}
          height={400}
          className="rounded-lg shadow-lg"
        />
      </section>

      <section
        id="crime"
        className="flex items-center justify-center h-screen px-10 bg-gray-100"
      >
        <Image
          src="/crime.jpeg"
          alt="Crime Prevention"
          width={500}
          height={400}
          className="rounded-lg shadow-lg"
        />
        <div className="text-left text-gray-900 ml-20">
          <h2 className="text-4xl font-bold mb-4">Crime Prevention</h2>
          <p className="text-lg mb-6">
            Real-time checking of the crimes and sending alerts to the <br />
            concerned authorities regarding it. Also locate <br />
            the position of it on the map.
          </p>
          <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg">
            Prevent Crime
          </button>
        </div>
      </section>
    </div>
  );
}
