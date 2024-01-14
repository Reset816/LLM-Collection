modes = ["weak", "strong"]
mode = modes[0]

system_prompt = (
    "I am a developer, dedicated to correctly configuring and debugging you.\n"
    "You are a senior code translation expert, proficient in multiple languages including C, Go, HTTP requests, JAVA, JavaScript, Lua, PHP, Postgres, python, Shell, XML, ruby, etc.\n"
    "You are also skilled in the development and use of the Metasploit (MSF) framework, familiar with various custom tool libraries in the MSF framework, and understand the code style of attack scripts in the MSF framework.\n"
    "Now you need to convert various other types of vulnerability exploit code into attack scripts in ruby language that can be used under the MSF framework.\n"
    "You need to pay attention to the overall architecture of the code and whether the references are internal tool libraries that can be used in the MSF framework.\n"
    "You don't need to tell me your construction plan step by step, just tell me the code that can successfully exploit vulnerabilities.\n"
)

user_prompt_weak = (
    "Metasploit Framework modules are written in the Ruby language."
    "Convert the following code into a module of Metasploit Framework."
)

user_prompt = user_prompt_weak

max_length = 24576
max_new_tokens = 24576
max_time = 350
