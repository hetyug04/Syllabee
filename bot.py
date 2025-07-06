import discord
from discord.ext import commands
import os
import aiohttp

import logging

logger = logging.getLogger(__name__)

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="/", intents=intents, application_id=os.getenv("DISCORD_APP_ID"))



API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

class ClarificationButton(discord.ui.Button):
    def __init__(self, course: dict):
        super().__init__(
            label=course.get("course_code"),
            style=discord.ButtonStyle.secondary,
            custom_id=f"clarify_select_{course.get('id')}"
        )
        self.course_id = course.get("id")

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)

        original_interaction = self.view.original_interaction
        original_question = self.view.original_question
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "user_id": str(original_interaction.user.id),
                "query": original_question,
                "syllabus_id": self.course_id 
            }
            async with session.post(f'{API_URL}/ask', json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    answer = data.get("answer", "I couldn't find an answer for that course.")
                    
                    embed = discord.Embed(title=f"❓ Your Question about {self.label}", description=f"```{original_question}```", color=discord.Color.blue())
                    embed.add_field(name="��� Answer", value=answer, inline=False)
                    
                    await original_interaction.followup.send(embed=embed)
                else:
                    await original_interaction.followup.send("Sorry, something went wrong when I tried to re-ask your question.")

        for child in self.view.children:
            child.disabled = True
        await original_interaction.edit_original_response(view=self.view)
        self.view.stop()



class ClarificationView(discord.ui.View):
    def __init__(self, courses: list, original_interaction: discord.Interaction, question: str):
        super().__init__(timeout=180)
        self.original_interaction = original_interaction
        self.original_question = question
        
        for course in courses:
            self.add_item(ClarificationButton(course=course))


@bot.event
async def on_ready():
    logger.info(f"Logged in as {bot.user}")
    logger.info("Attempting to sync global commands...")
    try:
        await bot.tree.sync()
        logger.info("Global commands sync initiated.")
    except Exception as e:
        logger.error(f"Failed to sync global commands: {e}")


@bot.tree.command(name="upload", description="Upload a syllabus PDF")
async def upload(interaction: discord.Interaction, file: discord.Attachment):
    await interaction.response.defer()
    
    logger.info(f"Received upload command for course from user {interaction.user.id}")
    
    try:
        async with aiohttp.ClientSession() as session:
            file_bytes = await file.read()
            
            form_data = aiohttp.FormData()
            form_data.add_field('file', file_bytes, filename=file.filename, content_type=file.content_type)
            form_data.add_field('user_id', str(interaction.user.id))

            logger.info(f"Sending file '{file.filename}' to FastAPI backend...")
            async with session.post(f'{API_URL}/upload', data=form_data) as response:
                response_text = await response.text()
                logger.info(f"FastAPI response ({response.status})")
                if response.status == 200:
                    
                    data = await response.json()
                    det = data["details"]
                    school = det["school"]
                    course = det["course_code"]
                    prof = det["professor"]
                    sem = det["semester"]
                    new_sub = det["is_new_subscription"]
                    new_syn = det["is_new"]
                    
                    if not new_sub:
                        return await interaction.followup.send(
                            f"You are already subscribed to **{school} - {course}**, Professor {prof}, {sem}! You can use `/ask`."
                        )
                    if new_syn:
                        await interaction.followup.send(
                            f"Syllabus for **{school} - {course}** uploaded and processed successfully! You can now use `/ask`."
                        )
                    return await interaction.followup.send(
                        f"You have been subscribed to **{school} - {course}**, Professor {prof}, {sem} You can now use `/ask`"
                    )
                else:
                    try:
                        error_data = await response.json()
                        error_message = error_data.get("message", "An unknown error occurred.")
                    except aiohttp.ContentTypeError:
                        error_message = await response.text()

                    if response.status == 400:
                        user_friendly_message = f"It looks like there was an issue with the file you uploaded. The server said: '**{error_message}**'. Please check the file and try again."
                    elif response.status == 413:
                        user_friendly_message = f"The file you tried to upload is too large. The maximum size is 20MB. Please upload a smaller file."
                    elif response.status == 500:
                        user_friendly_message = "I encountered a problem on my end while trying to process your syllabus. My team has been notified. Please try again in a little while."
                    else:
                        user_friendly_message = f"An unexpected error occurred (Status: {response.status}). If this persists, please contact support."

                    logger.error(f"FastAPI response ({response.status}): {await response.text()}")
                    await interaction.followup.send(user_friendly_message)
    except Exception as e:
        logger.error(f"An error occurred during upload: {e}")
        await interaction.followup.send("An unexpected error occurred. Please try again later.")

@bot.tree.command(name="ask", description="Ask a question about your syllabi")
async def ask(interaction: discord.Interaction, question: str):
    await interaction.response.defer()
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"query": question, "user_id": str(interaction.user.id)}
            async with session.post(f'{API_URL}/ask', json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("clarification_needed"):
                        courses = data.get("courses", [])
                        embed = discord.Embed(
                            title="🤔 Which course are you asking about?",
                            description=data.get("clarification_message", "Please choose one of the following courses."),
                            color=discord.Color.gold()
                        )
                        view = ClarificationView(courses, interaction, question)
                        await interaction.followup.send(embed=embed, view=view)
                    else:
                        answer = data.get("answer", "I'm sorry, I couldn't find an answer.")
                        embed = discord.Embed(
                            title=f"❓ Your Question: {question}", 
                            description=answer,
                            color=discord.Color.blue()
                        )
                        await interaction.followup.send(embed=embed)
                else:
                    await interaction.followup.send(f"Error asking question: {response.status} {await response.text()}")
    except Exception as e:
        logger.error(f"An error occurred during ask: {e}")
        await interaction.followup.send("An unexpected error occurred. Please try again later.")




@bot.tree.command(name="list", description="List your subscribed courses")
async def list_courses(interaction: discord.Interaction):
    await interaction.response.defer()
    try:
        async with aiohttp.ClientSession() as session:
            params = {"user_id": str(interaction.user.id)}
            async with session.get(f'{API_URL}/list', params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    courses_by_school = data.get("courses_by_school", {})
                    if not courses_by_school:
                        await interaction.followup.send("You are not subscribed to any courses.")
                        return

                    embed = discord.Embed(title="Your Subscribed Courses", color=discord.Color.blue())
                    for school, semesters in courses_by_school.items():
                        school_content = ""
                        for semester, courses in semesters.items():
                            school_content += f"**{semester}**\n" + "\n".join(courses) + "\n"
                        embed.add_field(name=school, value=school_content, inline=False)
                    
                    await interaction.followup.send(embed=embed)
                else:
                    await interaction.followup.send(f"Error listing courses: {response.status} {await response.text()}")
    except Exception as e:
        logger.error(f"An error occurred during list: {e}")
        await interaction.followup.send("An unexpected error occurred. Please try again later.")

@bot.tree.command(name="list_all", description="List all syllabi in the system")
async def list_all_courses(interaction: discord.Interaction):
    await interaction.response.defer()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{API_URL}/list_all') as response:
                if response.status == 200:
                    data = await response.json()
                    courses_by_school_semester_subject = data.get("courses_by_school_semester_subject", {})
                    if not courses_by_school_semester_subject:
                        await interaction.followup.send("No syllabi in the system.")
                        return

                    embed = discord.Embed(title="All Syllabi", color=discord.Color.green())
                    for school, semesters in courses_by_school_semester_subject.items():
                        school_content = ""
                        for semester, subjects in semesters.items():
                            school_content += f"**{semester}**\n"
                            for subject, courses in subjects.items():
                                school_content += f"***{subject}***\n" + "\n".join(courses) + "\n"
                        embed.add_field(name=school, value=school_content, inline=False)

                    await interaction.followup.send(embed=embed)
                else:
                    await interaction.followup.send(f"Error listing all courses: {response.status} {await response.text()}")
    except Exception as e:
        logger.error(f"An error occurred during list_all: {e}")
        await interaction.followup.send("An unexpected error occurred. Please try again later.")



@bot.tree.command(name="subscribe", description="Subscribe to a syllabus")
async def subscribe(interaction: discord.Interaction, syllabus_id: int):
    await interaction.response.defer()
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"user_id": str(interaction.user.id), "syllabus_id": syllabus_id}
            async with session.post(f'{API_URL}/subscribe', json=payload) as response:
                data = await response.json()
                await interaction.followup.send(data.get("message"))
    except Exception as e:
        logger.error(f"An error occurred during subscribe: {e}")
        await interaction.followup.send("An unexpected error occurred. Please try again later.")


@bot.tree.command(name="help", description="Shows a list of all available commands.")
async def help(interaction: discord.Interaction):
    embed = discord.Embed(
        title="Syllabus Bot Help",
        description="Here is a list of all available commands:",
        color=discord.Color.purple()
    )

    for command in bot.tree.get_commands():
        params = "".join(f" <{param.name}>" for param in command.parameters)
        embed.add_field(name=f"/{command.name}{params}", value=command.description, inline=False)

    await interaction.response.send_message(embed=embed, ephemeral=True)




@bot.tree.command(name="unsubscribe", description="Unsubscribe from a syllabus")
async def unsubscribe(interaction: discord.Interaction, syllabus_id: int):
    await interaction.response.defer()
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"user_id": str(interaction.user.id), "syllabus_id": syllabus_id}
            async with session.post(f'{API_URL}/unsubscribe', json=payload) as response:
                data = await response.json()
                await interaction.followup.send(data.get("message"))
    except Exception as e:
        logger.error(f"An error occurred during unsubscribe: {e}")
        await interaction.followup.send("An unexpected error occurred. Please try again later.")

