prompts = {}

prompts["keywords_extraction"] = '''
Task:
Extract entities (e.g., names, dates, locations, organizations, products) and their types from the provided text. Format your response as a list of entities with their types in parentheses, separated by commas.
Examples:
Input Text:
"John Smith visited Paris in 2022 to attend the United Nations Climate Summit."
Output:
John Smith[Person], Paris [Location], 2022 [Date], United Nations [Organization], Climate Summit [Event]
Input Text:
"Apple Inc. launched the iPhone 14 in September 2023 at their headquarters in Cupertino, California."
Output:
Apple Inc. [Organization], iPhone 14 [Product], September 2023 [Date], Cupertino [Location], California [Location]
Input Text:
"Dr. Maria Alvarez will speak at the TED Conference in Vancouver on July 15, 2024."
Output:
Dr. Maria Alvarez [Person], TED Conference [Event], Vancouver [Location], July 15, 2024 [Date]
Your Task:
Apply the same extraction to the following text:
'''


prompts["triples_extraction"] = '''
Task:
Given a passage and a list of extracted entities, generate triples that describe relationships between the entities. Use the format:
[Entity 1] <Relation> [Entity 2]
Examples:
Passage:
"Elon Musk, CEO of Tesla, announced the Cybertruck launch in Austin, Texas, on November 22, 2023."
Entities:
Elon Musk, Tesla, Cybertruck, Austin, Texas, November 22, 2023
Output:
Elon Musk <works_for> Tesla
Tesla <headquartered_in> Austin
Tesla <launched> Cybertruck
Cybertruck <launched_in> Austin
Cybertruck <launch_date> November 22, 2023
Passage:
"Dr. Jane Goodall visited the Serengeti National Park in Tanzania in 2021 to study chimpanzee behavior."
Entities:
Dr. Jane Goodall, Serengeti National Park, Tanzania, 2021, chimpanzee behavior
Output:
Dr. Jane Goodall <visited> Serengeti National Park
Serengeti National Park <located_in> Tanzania
Dr. Jane Goodall <studied> chimpanzee behavior
Dr. Jane Goodall <visited_in> 2021
Passage:
"Microsoft released Windows 11 on October 5, 2021, and partnered with Dell to distribute it globally."
Entities:
Microsoft, Windows 11, October 5, 2021, Dell
Output:
Microsoft <released> Windows 11
Windows 11 <release_date> October 5, 2021
Microsoft <partnered_with> Dell
Dell <distributed> Windows 11
Your Task:
Generate triples for the following passage and entities:
Passage: "{content}"
Entities:
'''