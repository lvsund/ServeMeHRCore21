USE [master]
GO
/****** Object:  Database [ServeMeHRCore]    Script Date: 2018-06-11 10:01:07 AM ******/
CREATE DATABASE [ServeMeHRCore]
 CONTAINMENT = NONE
 ON  PRIMARY 
( NAME = N'ServeMeHRCore', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL14.VS2019\MSSQL\DATA\ServeMeHRCore.mdf' , SIZE = 73728KB , MAXSIZE = UNLIMITED, FILEGROWTH = 65536KB )
 LOG ON 
( NAME = N'ServeMeHRCore_log', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL14.VS2019\MSSQL\DATA\ServeMeHRCore_log.ldf' , SIZE = 8192KB , MAXSIZE = 2048GB , FILEGROWTH = 65536KB )
GO
ALTER DATABASE [ServeMeHRCore] SET COMPATIBILITY_LEVEL = 140
GO
IF (1 = FULLTEXTSERVICEPROPERTY('IsFullTextInstalled'))
begin
EXEC [ServeMeHRCore].[dbo].[sp_fulltext_database] @action = 'enable'
end
GO
ALTER DATABASE [ServeMeHRCore] SET ANSI_NULL_DEFAULT OFF 
GO
ALTER DATABASE [ServeMeHRCore] SET ANSI_NULLS OFF 
GO
ALTER DATABASE [ServeMeHRCore] SET ANSI_PADDING OFF 
GO
ALTER DATABASE [ServeMeHRCore] SET ANSI_WARNINGS OFF 
GO
ALTER DATABASE [ServeMeHRCore] SET ARITHABORT OFF 
GO
ALTER DATABASE [ServeMeHRCore] SET AUTO_CLOSE OFF 
GO
ALTER DATABASE [ServeMeHRCore] SET AUTO_SHRINK OFF 
GO
ALTER DATABASE [ServeMeHRCore] SET AUTO_UPDATE_STATISTICS ON 
GO
ALTER DATABASE [ServeMeHRCore] SET CURSOR_CLOSE_ON_COMMIT OFF 
GO
ALTER DATABASE [ServeMeHRCore] SET CURSOR_DEFAULT  GLOBAL 
GO
ALTER DATABASE [ServeMeHRCore] SET CONCAT_NULL_YIELDS_NULL OFF 
GO
ALTER DATABASE [ServeMeHRCore] SET NUMERIC_ROUNDABORT OFF 
GO
ALTER DATABASE [ServeMeHRCore] SET QUOTED_IDENTIFIER OFF 
GO
ALTER DATABASE [ServeMeHRCore] SET RECURSIVE_TRIGGERS OFF 
GO
ALTER DATABASE [ServeMeHRCore] SET  ENABLE_BROKER 
GO
ALTER DATABASE [ServeMeHRCore] SET AUTO_UPDATE_STATISTICS_ASYNC OFF 
GO
ALTER DATABASE [ServeMeHRCore] SET DATE_CORRELATION_OPTIMIZATION OFF 
GO
ALTER DATABASE [ServeMeHRCore] SET TRUSTWORTHY OFF 
GO
ALTER DATABASE [ServeMeHRCore] SET ALLOW_SNAPSHOT_ISOLATION OFF 
GO
ALTER DATABASE [ServeMeHRCore] SET PARAMETERIZATION SIMPLE 
GO
ALTER DATABASE [ServeMeHRCore] SET READ_COMMITTED_SNAPSHOT ON 
GO
ALTER DATABASE [ServeMeHRCore] SET HONOR_BROKER_PRIORITY OFF 
GO
ALTER DATABASE [ServeMeHRCore] SET RECOVERY FULL 
GO
ALTER DATABASE [ServeMeHRCore] SET  MULTI_USER 
GO
ALTER DATABASE [ServeMeHRCore] SET PAGE_VERIFY CHECKSUM  
GO
ALTER DATABASE [ServeMeHRCore] SET DB_CHAINING OFF 
GO
ALTER DATABASE [ServeMeHRCore] SET FILESTREAM( NON_TRANSACTED_ACCESS = OFF ) 
GO
ALTER DATABASE [ServeMeHRCore] SET TARGET_RECOVERY_TIME = 60 SECONDS 
GO
ALTER DATABASE [ServeMeHRCore] SET DELAYED_DURABILITY = DISABLED 
GO
EXEC sys.sp_db_vardecimal_storage_format N'ServeMeHRCore', N'ON'
GO
ALTER DATABASE [ServeMeHRCore] SET QUERY_STORE = OFF
GO
USE [ServeMeHRCore]
GO
ALTER DATABASE SCOPED CONFIGURATION SET IDENTITY_CACHE = ON;
GO
ALTER DATABASE SCOPED CONFIGURATION SET LEGACY_CARDINALITY_ESTIMATION = OFF;
GO
ALTER DATABASE SCOPED CONFIGURATION FOR SECONDARY SET LEGACY_CARDINALITY_ESTIMATION = PRIMARY;
GO
ALTER DATABASE SCOPED CONFIGURATION SET MAXDOP = 0;
GO
ALTER DATABASE SCOPED CONFIGURATION FOR SECONDARY SET MAXDOP = PRIMARY;
GO
ALTER DATABASE SCOPED CONFIGURATION SET PARAMETER_SNIFFING = ON;
GO
ALTER DATABASE SCOPED CONFIGURATION FOR SECONDARY SET PARAMETER_SNIFFING = PRIMARY;
GO
ALTER DATABASE SCOPED CONFIGURATION SET QUERY_OPTIMIZER_HOTFIXES = OFF;
GO
ALTER DATABASE SCOPED CONFIGURATION FOR SECONDARY SET QUERY_OPTIMIZER_HOTFIXES = PRIMARY;
GO
USE [ServeMeHRCore]
GO
/****** Object:  Table [dbo].[ADInformations]    Script Date: 2018-06-11 10:01:08 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[ADInformations](
	[Id] [int] IDENTITY(1,1) NOT NULL,
	[sAMAccountNAme] [nvarchar](255) NOT NULL,
	[displayName] [nvarchar](255) NULL,
	[mail] [nvarchar](255) NULL,
	[title] [nvarchar](255) NULL,
	[telephoneNumber] [nvarchar](255) NULL,
	[givenName] [nvarchar](255) NOT NULL,
	[sn] [nvarchar](255) NOT NULL,
	[company] [nvarchar](255) NULL,
	[wwWHomePage] [nvarchar](255) NULL,
	[mobile] [nvarchar](255) NULL,
	[cn] [nvarchar](255) NULL,
	[APPUSERNAME] [nvarchar](255) NOT NULL,
 CONSTRAINT [PK_ADInformations] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[ApplicConfs]    Script Date: 2018-06-11 10:01:08 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[ApplicConfs](
	[Id] [int] IDENTITY(1,1) NOT NULL,
	[FileSystemUpload] [bit] NOT NULL,
	[ADActive] [bit] NOT NULL,
	[EmailConfirmation] [bit] NOT NULL,
	[ModifiedBy] [nvarchar](255) NULL,
	[Modified] [datetimeoffset](7) NULL,
	[AppAdmin] [nvarchar](255) NULL,
	[BackAdmin] [nvarchar](255) NULL,
	[LDAPConn] [nvarchar](255) NULL,
	[LDAPPath] [nvarchar](255) NULL,
	[ManageHREmail] [nvarchar](255) NULL,
	[ManageHREmailPass] [nvarchar](255) NULL,
	[SMTPHost] [nvarchar](255) NULL,
	[SMTPPort] [int] NULL,
	[EnableSSL] [bit] NULL,
 CONSTRAINT [PK_ApplicConfs] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Appointments]    Script Date: 2018-06-11 10:01:08 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Appointments](
	[Id] [int] IDENTITY(1,1) NOT NULL,
	[Subject] [nvarchar](255) NOT NULL,
	[StartTime] [datetime] NOT NULL,
	[EndTime] [datetime] NOT NULL,
	[Location] [nvarchar](255) NULL,
	[Notes] [nvarchar](255) NULL,
	[MsgID] [nvarchar](255) NULL,
	[MsgSequence] [int] NULL,
	[SenderEmail] [nvarchar](255) NOT NULL,
	[RecipientEmail] [nvarchar](255) NOT NULL,
 CONSTRAINT [PK_Appointments] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[FileDetails]    Script Date: 2018-06-11 10:01:08 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[FileDetails](
	[Id] [uniqueidentifier] NOT NULL,
	[FileName] [nvarchar](255) NULL,
	[Extension] [nvarchar](10) NULL,
	[ServiceRequestID] [int] NULL,
 CONSTRAINT [PK_FileDetails] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[IndividualAssignmentHistories]    Script Date: 2018-06-11 10:01:08 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[IndividualAssignmentHistories](
	[Id] [int] IDENTITY(1,1) NOT NULL,
	[AssignedTo] [int] NOT NULL,
	[AssignedBy] [nvarchar](255) NOT NULL,
	[DateAssigned] [datetime] NOT NULL,
	[ServiceRequest] [int] NULL,
 CONSTRAINT [PK_IndividualAssignmentHistories] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Members]    Script Date: 2018-06-11 10:01:08 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Members](
	[Id] [int] IDENTITY(1,1) NOT NULL,
	[MemberUserid] [nvarchar](255) NOT NULL,
	[MemberFirstName] [nvarchar](255) NOT NULL,
	[MemberLastName] [nvarchar](255) NOT NULL,
	[MemberFullName] [nvarchar](255) NOT NULL,
	[MemberEmail] [nvarchar](255) NOT NULL,
	[MemberPhone] [nvarchar](255) NOT NULL,
 CONSTRAINT [PK_Members] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Priorities]    Script Date: 2018-06-11 10:01:08 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Priorities](
	[Id] [int] IDENTITY(1,1) NOT NULL,
	[PriorityDescription] [nvarchar](255) NOT NULL,
	[LastUpdated] [datetime] NOT NULL,
	[Active] [bit] NOT NULL,
	[Team] [int] NOT NULL,
 CONSTRAINT [PK_Priorities] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[RequestTypes]    Script Date: 2018-06-11 10:01:08 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[RequestTypes](
	[Id] [int] IDENTITY(1,1) NOT NULL,
	[RequestTypeDescription] [nvarchar](50) NOT NULL,
	[LastUpdated] [datetime] NULL,
	[Active] [bit] NULL,
	[Team] [int] NOT NULL,
 CONSTRAINT [PK_RequestTypes] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[RequestTypeSteps]    Script Date: 2018-06-11 10:01:08 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[RequestTypeSteps](
	[Id] [int] IDENTITY(1,1) NOT NULL,
	[StepDescription] [nvarchar](255) NOT NULL,
	[StepNumber] [int] NOT NULL,
	[LastUpdated] [datetime] NULL,
	[Active] [bit] NULL,
	[RequestType] [int] NOT NULL,
 CONSTRAINT [PK_RequestTypeSteps] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[ServiceRequestNotes]    Script Date: 2018-06-11 10:01:08 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[ServiceRequestNotes](
	[Id] [int] IDENTITY(1,1) NOT NULL,
	[NoteDescription] [nvarchar](255) NOT NULL,
	[LastUpdated] [datetime] NULL,
	[WrittenBy] [nvarchar](255) NULL,
	[ServiceRequest] [int] NULL,
 CONSTRAINT [PK_ServiceRequestNotes] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[ServiceRequests]    Script Date: 2018-06-11 10:01:08 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[ServiceRequests](
	[Id] [int] IDENTITY(1,1) NOT NULL,
	[RequestHeading] [nvarchar](255) NOT NULL,
	[RequestDescription] [nvarchar](max) NOT NULL,
	[RequestorID] [nvarchar](255) NOT NULL,
	[RequestorFirstName] [nvarchar](255) NULL,
	[RequestorLastName] [nvarchar](255) NULL,
	[RequestorPhone] [nvarchar](255) NULL,
	[RequestorEmail] [nvarchar](255) NOT NULL,
	[DateTimeSubmitted] [datetime] NULL,
	[DateTimeStarted] [datetime] NULL,
	[DateTimeCompleted] [datetime] NULL,
	[Priority] [int] NULL,
	[RequestType] [int] NULL,
	[RequestTypeStep] [int] NULL,
	[Member] [int] NULL,
	[Status] [int] NOT NULL,
	[Team] [int] NOT NULL,
 CONSTRAINT [PK_ServiceRequests] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[StatusSets]    Script Date: 2018-06-11 10:01:08 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[StatusSets](
	[Id] [int] IDENTITY(1,1) NOT NULL,
	[StatusDescription] [nvarchar](255) NOT NULL,
	[LastUpdated] [datetime] NOT NULL,
	[Active] [bit] NOT NULL,
	[StatusType] [int] NOT NULL,
 CONSTRAINT [PK_StatusSets] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[StatusTypes]    Script Date: 2018-06-11 10:01:08 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[StatusTypes](
	[Id] [int] IDENTITY(1,1) NOT NULL,
	[StatusTypeDescription] [nvarchar](255) NOT NULL,
 CONSTRAINT [PK_StatusTypes] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[StepHistories]    Script Date: 2018-06-11 10:01:08 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[StepHistories](
	[Id] [int] IDENTITY(1,1) NOT NULL,
	[LastUpdated] [datetime] NOT NULL,
	[RequestTypeStep] [int] NOT NULL,
	[ServiceRequest] [int] NULL,
 CONSTRAINT [PK_StepHistories] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[TeamAssignmentHistories]    Script Date: 2018-06-11 10:01:08 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[TeamAssignmentHistories](
	[Id] [int] IDENTITY(1,1) NOT NULL,
	[AssignedBy] [nvarchar](255) NOT NULL,
	[DateAssigned] [datetime] NOT NULL,
	[ServiceRequest] [int] NOT NULL,
	[Team] [int] NOT NULL,
 CONSTRAINT [PK_TeamAssignmentHistories] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[TeamMembers]    Script Date: 2018-06-11 10:01:08 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[TeamMembers](
	[Id] [int] IDENTITY(1,1) NOT NULL,
	[Member] [int] NOT NULL,
	[Team] [int] NOT NULL,
 CONSTRAINT [PK_TeamMembers] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Teams]    Script Date: 2018-06-11 10:01:08 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Teams](
	[Id] [int] IDENTITY(1,1) NOT NULL,
	[TeamDescription] [nvarchar](255) NOT NULL,
	[TeamEmailAddress] [nvarchar](255) NOT NULL,
 CONSTRAINT [PK_Teams] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO
SET IDENTITY_INSERT [dbo].[ApplicConfs] ON 

INSERT [dbo].[ApplicConfs] ([Id], [FileSystemUpload], [ADActive], [EmailConfirmation], [ModifiedBy], [Modified], [AppAdmin], [BackAdmin], [LDAPConn], [LDAPPath], [ManageHREmail], [ManageHREmailPass], [SMTPHost], [SMTPPort], [EnableSSL]) VALUES (1, 1, 1, 1, N'A3HR\Lyndon', CAST(N'2017-11-18T15:59:51.2350181-08:00' AS DateTimeOffset), N'A3HR\Lyndon', N'A3HR\Lyndon', N'SERVER.A3HR.local', N'LDAP://SERVER.A3HR.local', N'', N'', N'smtp.gmail.com', 587, 1)
SET IDENTITY_INSERT [dbo].[ApplicConfs] OFF
SET IDENTITY_INSERT [dbo].[IndividualAssignmentHistories] ON 

INSERT [dbo].[IndividualAssignmentHistories] ([Id], [AssignedTo], [AssignedBy], [DateAssigned], [ServiceRequest]) VALUES (3, 1, N'A3HR\Lyndon', CAST(N'2018-05-11T15:18:29.793' AS DateTime), 7)
INSERT [dbo].[IndividualAssignmentHistories] ([Id], [AssignedTo], [AssignedBy], [DateAssigned], [ServiceRequest]) VALUES (4, 2, N'A3HR\Lyndon', CAST(N'2018-05-11T15:53:40.523' AS DateTime), 7)
INSERT [dbo].[IndividualAssignmentHistories] ([Id], [AssignedTo], [AssignedBy], [DateAssigned], [ServiceRequest]) VALUES (5, 1, N'A3HR\Lyndon', CAST(N'2018-05-11T15:54:01.493' AS DateTime), 7)
INSERT [dbo].[IndividualAssignmentHistories] ([Id], [AssignedTo], [AssignedBy], [DateAssigned], [ServiceRequest]) VALUES (6, 2, N'A3HR\Lyndon', CAST(N'2018-05-11T18:21:13.763' AS DateTime), 7)
INSERT [dbo].[IndividualAssignmentHistories] ([Id], [AssignedTo], [AssignedBy], [DateAssigned], [ServiceRequest]) VALUES (8, 2, N'A3HR\Lyndon', CAST(N'2018-05-11T18:50:30.693' AS DateTime), 9)
INSERT [dbo].[IndividualAssignmentHistories] ([Id], [AssignedTo], [AssignedBy], [DateAssigned], [ServiceRequest]) VALUES (10, 1, N'A3HR\Lyndon', CAST(N'2018-05-12T07:53:56.903' AS DateTime), 11)
INSERT [dbo].[IndividualAssignmentHistories] ([Id], [AssignedTo], [AssignedBy], [DateAssigned], [ServiceRequest]) VALUES (11, 1, N'A3HR\Lyndon', CAST(N'2018-05-12T14:12:48.957' AS DateTime), 12)
INSERT [dbo].[IndividualAssignmentHistories] ([Id], [AssignedTo], [AssignedBy], [DateAssigned], [ServiceRequest]) VALUES (14, 1, N'A3HR\Lyndon', CAST(N'2018-05-19T10:21:32.880' AS DateTime), 7)
INSERT [dbo].[IndividualAssignmentHistories] ([Id], [AssignedTo], [AssignedBy], [DateAssigned], [ServiceRequest]) VALUES (15, 1, N'A3HR\Lyndon', CAST(N'2018-05-19T10:22:14.853' AS DateTime), 9)
INSERT [dbo].[IndividualAssignmentHistories] ([Id], [AssignedTo], [AssignedBy], [DateAssigned], [ServiceRequest]) VALUES (16, 1, N'A3HR\Lyndon', CAST(N'2018-05-19T14:12:28.243' AS DateTime), 22)
INSERT [dbo].[IndividualAssignmentHistories] ([Id], [AssignedTo], [AssignedBy], [DateAssigned], [ServiceRequest]) VALUES (17, 1, N'A3HR\Lyndon', CAST(N'2018-05-19T14:38:31.377' AS DateTime), 24)
INSERT [dbo].[IndividualAssignmentHistories] ([Id], [AssignedTo], [AssignedBy], [DateAssigned], [ServiceRequest]) VALUES (18, 1, N'A3HR\Lyndon', CAST(N'2018-05-19T14:45:01.230' AS DateTime), 25)
INSERT [dbo].[IndividualAssignmentHistories] ([Id], [AssignedTo], [AssignedBy], [DateAssigned], [ServiceRequest]) VALUES (19, 1, N'A3HR\Lyndon', CAST(N'2018-05-19T14:58:05.630' AS DateTime), 26)
INSERT [dbo].[IndividualAssignmentHistories] ([Id], [AssignedTo], [AssignedBy], [DateAssigned], [ServiceRequest]) VALUES (21, 1, N'A3HR\Lyndon', CAST(N'2018-05-20T13:47:03.270' AS DateTime), 28)
INSERT [dbo].[IndividualAssignmentHistories] ([Id], [AssignedTo], [AssignedBy], [DateAssigned], [ServiceRequest]) VALUES (22, 1, N'A3HR\Lyndon', CAST(N'2018-05-20T14:02:40.767' AS DateTime), 29)
INSERT [dbo].[IndividualAssignmentHistories] ([Id], [AssignedTo], [AssignedBy], [DateAssigned], [ServiceRequest]) VALUES (23, 1, N'A3HR\Lyndon', CAST(N'2018-05-20T14:04:18.840' AS DateTime), 30)
INSERT [dbo].[IndividualAssignmentHistories] ([Id], [AssignedTo], [AssignedBy], [DateAssigned], [ServiceRequest]) VALUES (24, 1, N'A3HR\Lyndon', CAST(N'2018-05-20T14:05:47.040' AS DateTime), 31)
INSERT [dbo].[IndividualAssignmentHistories] ([Id], [AssignedTo], [AssignedBy], [DateAssigned], [ServiceRequest]) VALUES (25, 1, N'A3HR\Lyndon', CAST(N'2018-05-20T14:06:41.973' AS DateTime), 32)
INSERT [dbo].[IndividualAssignmentHistories] ([Id], [AssignedTo], [AssignedBy], [DateAssigned], [ServiceRequest]) VALUES (26, 1, N'A3HR\Lyndon', CAST(N'2018-05-20T14:13:58.520' AS DateTime), 33)
INSERT [dbo].[IndividualAssignmentHistories] ([Id], [AssignedTo], [AssignedBy], [DateAssigned], [ServiceRequest]) VALUES (27, 1, N'A3HR\Lyndon', CAST(N'2018-05-20T14:26:01.073' AS DateTime), 34)
INSERT [dbo].[IndividualAssignmentHistories] ([Id], [AssignedTo], [AssignedBy], [DateAssigned], [ServiceRequest]) VALUES (28, 1, N'A3HR\Lyndon', CAST(N'2018-05-20T14:52:22.013' AS DateTime), 35)
INSERT [dbo].[IndividualAssignmentHistories] ([Id], [AssignedTo], [AssignedBy], [DateAssigned], [ServiceRequest]) VALUES (29, 1, N'A3HR\Lyndon', CAST(N'2018-05-20T14:53:55.987' AS DateTime), 36)
INSERT [dbo].[IndividualAssignmentHistories] ([Id], [AssignedTo], [AssignedBy], [DateAssigned], [ServiceRequest]) VALUES (30, 1, N'A3HR\Lyndon', CAST(N'2018-05-20T14:56:31.990' AS DateTime), 37)
INSERT [dbo].[IndividualAssignmentHistories] ([Id], [AssignedTo], [AssignedBy], [DateAssigned], [ServiceRequest]) VALUES (31, 1, N'A3HR\Lyndon', CAST(N'2018-05-25T11:07:46.027' AS DateTime), 39)
INSERT [dbo].[IndividualAssignmentHistories] ([Id], [AssignedTo], [AssignedBy], [DateAssigned], [ServiceRequest]) VALUES (32, 1, N'A3HR\Lyndon', CAST(N'2018-05-26T09:59:53.830' AS DateTime), 42)
INSERT [dbo].[IndividualAssignmentHistories] ([Id], [AssignedTo], [AssignedBy], [DateAssigned], [ServiceRequest]) VALUES (33, 1, N'A3HR\Lyndon', CAST(N'2018-05-26T10:07:31.443' AS DateTime), 44)
INSERT [dbo].[IndividualAssignmentHistories] ([Id], [AssignedTo], [AssignedBy], [DateAssigned], [ServiceRequest]) VALUES (34, 2, N'A3HR\Lyndon', CAST(N'2018-05-26T10:10:09.107' AS DateTime), 45)
INSERT [dbo].[IndividualAssignmentHistories] ([Id], [AssignedTo], [AssignedBy], [DateAssigned], [ServiceRequest]) VALUES (35, 2, N'A3HR\Lyndon', CAST(N'2018-05-26T11:44:08.567' AS DateTime), 32)
SET IDENTITY_INSERT [dbo].[IndividualAssignmentHistories] OFF
SET IDENTITY_INSERT [dbo].[Members] ON 

INSERT [dbo].[Members] ([Id], [MemberUserid], [MemberFirstName], [MemberLastName], [MemberFullName], [MemberEmail], [MemberPhone]) VALUES (1, N'UnAssigned', N'UnAssigned', N'UnAssigned', N'UnAssigned', N'Unassigned@Unassigned.com', N'000 000 0000')
INSERT [dbo].[Members] ([Id], [MemberUserid], [MemberFirstName], [MemberLastName], [MemberFullName], [MemberEmail], [MemberPhone]) VALUES (2, N'A3HR\Lyndon', N'Lyndon', N'Sundmark', N'Lyndon Sundmark', N'lvsund@outlook.com', N'604 202 4261')
SET IDENTITY_INSERT [dbo].[Members] OFF
SET IDENTITY_INSERT [dbo].[Priorities] ON 

INSERT [dbo].[Priorities] ([Id], [PriorityDescription], [LastUpdated], [Active], [Team]) VALUES (3, N'--UnAssigned', CAST(N'2017-05-12T12:00:00.000' AS DateTime), 1, 2)
INSERT [dbo].[Priorities] ([Id], [PriorityDescription], [LastUpdated], [Active], [Team]) VALUES (4, N'--UnAssigned', CAST(N'2018-05-26T12:00:00.000' AS DateTime), 1, 1)
INSERT [dbo].[Priorities] ([Id], [PriorityDescription], [LastUpdated], [Active], [Team]) VALUES (5, N'1-Low', CAST(N'2018-05-26T10:45:18.207' AS DateTime), 1, 2)
SET IDENTITY_INSERT [dbo].[Priorities] OFF
SET IDENTITY_INSERT [dbo].[RequestTypes] ON 

INSERT [dbo].[RequestTypes] ([Id], [RequestTypeDescription], [LastUpdated], [Active], [Team]) VALUES (3, N'--UnAssigned--', CAST(N'2018-05-12T12:00:00.000' AS DateTime), 1, 2)
INSERT [dbo].[RequestTypes] ([Id], [RequestTypeDescription], [LastUpdated], [Active], [Team]) VALUES (4, N'--Unassigned', CAST(N'2018-05-26T12:00:00.000' AS DateTime), 1, 1)
INSERT [dbo].[RequestTypes] ([Id], [RequestTypeDescription], [LastUpdated], [Active], [Team]) VALUES (5, N'Reports', CAST(N'2018-05-26T10:53:11.747' AS DateTime), 1, 2)
SET IDENTITY_INSERT [dbo].[RequestTypes] OFF
SET IDENTITY_INSERT [dbo].[RequestTypeSteps] ON 

INSERT [dbo].[RequestTypeSteps] ([Id], [StepDescription], [StepNumber], [LastUpdated], [Active], [RequestType]) VALUES (2, N'--Start--', 1, CAST(N'2018-05-12T12:00:00.000' AS DateTime), 1, 3)
INSERT [dbo].[RequestTypeSteps] ([Id], [StepDescription], [StepNumber], [LastUpdated], [Active], [RequestType]) VALUES (3, N'Gather Requirements', 1, CAST(N'2018-05-26T10:54:30.710' AS DateTime), 1, 5)
SET IDENTITY_INSERT [dbo].[RequestTypeSteps] OFF
SET IDENTITY_INSERT [dbo].[ServiceRequestNotes] ON 

INSERT [dbo].[ServiceRequestNotes] ([Id], [NoteDescription], [LastUpdated], [WrittenBy], [ServiceRequest]) VALUES (1, N'vbvbvbvbvbv', NULL, NULL, 7)
INSERT [dbo].[ServiceRequestNotes] ([Id], [NoteDescription], [LastUpdated], [WrittenBy], [ServiceRequest]) VALUES (2, N'xcxcxcx', CAST(N'2018-05-17T14:29:44.420' AS DateTime), N'A3HR\Lyndon', 7)
INSERT [dbo].[ServiceRequestNotes] ([Id], [NoteDescription], [LastUpdated], [WrittenBy], [ServiceRequest]) VALUES (3, N'fsfsfsfsf', CAST(N'2018-05-19T14:13:33.050' AS DateTime), N'A3HR\Lyndon', 22)
INSERT [dbo].[ServiceRequestNotes] ([Id], [NoteDescription], [LastUpdated], [WrittenBy], [ServiceRequest]) VALUES (4, N'cbcbc', CAST(N'2018-05-20T15:10:06.253' AS DateTime), N'A3HR\Lyndon', 37)
INSERT [dbo].[ServiceRequestNotes] ([Id], [NoteDescription], [LastUpdated], [WrittenBy], [ServiceRequest]) VALUES (5, N'cbcbcb', CAST(N'2018-05-20T15:10:49.370' AS DateTime), N'A3HR\Lyndon', 37)
INSERT [dbo].[ServiceRequestNotes] ([Id], [NoteDescription], [LastUpdated], [WrittenBy], [ServiceRequest]) VALUES (7, N'scsscsc', CAST(N'2018-05-20T15:16:42.410' AS DateTime), N'A3HR\Lyndon', 37)
INSERT [dbo].[ServiceRequestNotes] ([Id], [NoteDescription], [LastUpdated], [WrittenBy], [ServiceRequest]) VALUES (8, N'rhrhrhrhr', CAST(N'2018-05-20T15:20:44.123' AS DateTime), N'A3HR\Lyndon', 37)
INSERT [dbo].[ServiceRequestNotes] ([Id], [NoteDescription], [LastUpdated], [WrittenBy], [ServiceRequest]) VALUES (9, N'dvdvdvdv', CAST(N'2018-05-20T15:42:07.820' AS DateTime), N'A3HR\Lyndon', 37)
INSERT [dbo].[ServiceRequestNotes] ([Id], [NoteDescription], [LastUpdated], [WrittenBy], [ServiceRequest]) VALUES (10, N'fhfhfhh', CAST(N'2018-05-20T15:44:11.610' AS DateTime), N'A3HR\Lyndon', 37)
INSERT [dbo].[ServiceRequestNotes] ([Id], [NoteDescription], [LastUpdated], [WrittenBy], [ServiceRequest]) VALUES (11, N'xvxvxv', CAST(N'2018-05-20T19:04:55.077' AS DateTime), N'A3HR\Lyndon', 37)
INSERT [dbo].[ServiceRequestNotes] ([Id], [NoteDescription], [LastUpdated], [WrittenBy], [ServiceRequest]) VALUES (12, N'cbcbcb', CAST(N'2018-05-25T07:34:22.380' AS DateTime), N'A3HR\Lyndon', 7)
INSERT [dbo].[ServiceRequestNotes] ([Id], [NoteDescription], [LastUpdated], [WrittenBy], [ServiceRequest]) VALUES (13, N'cbcbcb', CAST(N'2018-05-25T09:31:34.177' AS DateTime), N'A3HR\Lyndon', 7)
INSERT [dbo].[ServiceRequestNotes] ([Id], [NoteDescription], [LastUpdated], [WrittenBy], [ServiceRequest]) VALUES (14, N'hffhhhfh', CAST(N'2018-05-25T09:33:14.270' AS DateTime), N'A3HR\Lyndon', 37)
INSERT [dbo].[ServiceRequestNotes] ([Id], [NoteDescription], [LastUpdated], [WrittenBy], [ServiceRequest]) VALUES (15, N'fhfhfhfh', CAST(N'2018-05-25T09:33:42.483' AS DateTime), N'A3HR\Lyndon', 37)
INSERT [dbo].[ServiceRequestNotes] ([Id], [NoteDescription], [LastUpdated], [WrittenBy], [ServiceRequest]) VALUES (17, N'xvxvxv', CAST(N'2018-05-26T10:04:19.463' AS DateTime), N'A3HR\Lyndon', 42)
INSERT [dbo].[ServiceRequestNotes] ([Id], [NoteDescription], [LastUpdated], [WrittenBy], [ServiceRequest]) VALUES (18, N'vnvnvv', CAST(N'2018-05-26T10:08:15.723' AS DateTime), N'A3HR\Lyndon', 44)
INSERT [dbo].[ServiceRequestNotes] ([Id], [NoteDescription], [LastUpdated], [WrittenBy], [ServiceRequest]) VALUES (19, N'vxvxvxv', CAST(N'2018-05-26T10:10:56.373' AS DateTime), N'A3HR\Lyndon', 45)
INSERT [dbo].[ServiceRequestNotes] ([Id], [NoteDescription], [LastUpdated], [WrittenBy], [ServiceRequest]) VALUES (20, N'dgddg', CAST(N'2018-05-26T10:17:10.750' AS DateTime), N'A3HR\Lyndon', 9)
INSERT [dbo].[ServiceRequestNotes] ([Id], [NoteDescription], [LastUpdated], [WrittenBy], [ServiceRequest]) VALUES (21, N'yryryryryry', CAST(N'2018-05-26T10:59:44.177' AS DateTime), N'A3HR\Lyndon', 7)
INSERT [dbo].[ServiceRequestNotes] ([Id], [NoteDescription], [LastUpdated], [WrittenBy], [ServiceRequest]) VALUES (22, N'dgdgd', CAST(N'2018-05-30T13:06:28.347' AS DateTime), N'A3HR\Lyndon', 45)
INSERT [dbo].[ServiceRequestNotes] ([Id], [NoteDescription], [LastUpdated], [WrittenBy], [ServiceRequest]) VALUES (23, N'jjgjggj', CAST(N'2018-05-30T14:00:09.407' AS DateTime), N'A3HR\Lyndon', 42)
INSERT [dbo].[ServiceRequestNotes] ([Id], [NoteDescription], [LastUpdated], [WrittenBy], [ServiceRequest]) VALUES (24, N'dgdgd', CAST(N'2018-05-30T14:00:28.207' AS DateTime), N'A3HR\Lyndon', 45)
INSERT [dbo].[ServiceRequestNotes] ([Id], [NoteDescription], [LastUpdated], [WrittenBy], [ServiceRequest]) VALUES (25, N'xvxvxv', CAST(N'2018-05-30T14:01:08.553' AS DateTime), N'A3HR\Lyndon', 45)
INSERT [dbo].[ServiceRequestNotes] ([Id], [NoteDescription], [LastUpdated], [WrittenBy], [ServiceRequest]) VALUES (26, N'khkhkh', CAST(N'2018-05-30T14:01:33.527' AS DateTime), N'A3HR\Lyndon', 45)
INSERT [dbo].[ServiceRequestNotes] ([Id], [NoteDescription], [LastUpdated], [WrittenBy], [ServiceRequest]) VALUES (27, N'tettt', CAST(N'2018-06-03T15:04:21.420' AS DateTime), N'A3HR\Lyndon', 45)
SET IDENTITY_INSERT [dbo].[ServiceRequestNotes] OFF
SET IDENTITY_INSERT [dbo].[ServiceRequests] ON 

INSERT [dbo].[ServiceRequests] ([Id], [RequestHeading], [RequestDescription], [RequestorID], [RequestorFirstName], [RequestorLastName], [RequestorPhone], [RequestorEmail], [DateTimeSubmitted], [DateTimeStarted], [DateTimeCompleted], [Priority], [RequestType], [RequestTypeStep], [Member], [Status], [Team]) VALUES (7, N'ngng gjgjjgj', N'gngnngng', N'A3HR\Admin', N'Lyndon', N'Sundmark', N'604 357 3669', N'lvsund@outlook.com', CAST(N'2018-05-11T15:18:29.790' AS DateTime), NULL, NULL, 3, 3, 2, 1, 1, 2)
INSERT [dbo].[ServiceRequests] ([Id], [RequestHeading], [RequestDescription], [RequestorID], [RequestorFirstName], [RequestorLastName], [RequestorPhone], [RequestorEmail], [DateTimeSubmitted], [DateTimeStarted], [DateTimeCompleted], [Priority], [RequestType], [RequestTypeStep], [Member], [Status], [Team]) VALUES (9, N'fghffh', N'fhfh gjgjgjgjg gfhfhfhfh', N'A3HR\Lyndon', N'Lyndon', N'Sundmark', N'604 357 3669', N'lvsund@outlook.com', NULL, CAST(N'2018-05-26T10:14:07.827' AS DateTime), NULL, 3, 3, 2, 1, 2, 2)
INSERT [dbo].[ServiceRequests] ([Id], [RequestHeading], [RequestDescription], [RequestorID], [RequestorFirstName], [RequestorLastName], [RequestorPhone], [RequestorEmail], [DateTimeSubmitted], [DateTimeStarted], [DateTimeCompleted], [Priority], [RequestType], [RequestTypeStep], [Member], [Status], [Team]) VALUES (11, N'xcxc', N'xcxc  cbcbbcb', N'A3HR\Lyndon', N'Lyndon', N'Sundmark', N'604 357 3669', N'lvsund@outlook.com', NULL, NULL, NULL, NULL, NULL, 2, 1, 1, 1)
INSERT [dbo].[ServiceRequests] ([Id], [RequestHeading], [RequestDescription], [RequestorID], [RequestorFirstName], [RequestorLastName], [RequestorPhone], [RequestorEmail], [DateTimeSubmitted], [DateTimeStarted], [DateTimeCompleted], [Priority], [RequestType], [RequestTypeStep], [Member], [Status], [Team]) VALUES (12, N'cbcbcb', N'cbcbc', N'A3HR\Lyndon', N'Lyndon', N'Sundmark', N'604 357 3669', N'lvsund@outlook.com', NULL, NULL, NULL, NULL, NULL, 2, 1, 1, 1)
INSERT [dbo].[ServiceRequests] ([Id], [RequestHeading], [RequestDescription], [RequestorID], [RequestorFirstName], [RequestorLastName], [RequestorPhone], [RequestorEmail], [DateTimeSubmitted], [DateTimeStarted], [DateTimeCompleted], [Priority], [RequestType], [RequestTypeStep], [Member], [Status], [Team]) VALUES (22, N'sfsfsfs', N'sfsfssfsfsfs', N'A3HR\Lyndon', N'Lyndon', N'Sundmark', N'604 357 3669', N'lvsund@outlook.com', CAST(N'2018-05-19T14:12:28.243' AS DateTime), NULL, NULL, 3, 3, 2, 1, 1, 1)
INSERT [dbo].[ServiceRequests] ([Id], [RequestHeading], [RequestDescription], [RequestorID], [RequestorFirstName], [RequestorLastName], [RequestorPhone], [RequestorEmail], [DateTimeSubmitted], [DateTimeStarted], [DateTimeCompleted], [Priority], [RequestType], [RequestTypeStep], [Member], [Status], [Team]) VALUES (24, N'hgjjgj', N'gjgj', N'A3HR\Lyndon', N'Lyndon', N'Sundmark', N'604 357 3669', N'lvsund@outlook.com', CAST(N'2018-05-19T14:38:31.377' AS DateTime), NULL, NULL, 3, 3, 2, 1, 1, 2)
INSERT [dbo].[ServiceRequests] ([Id], [RequestHeading], [RequestDescription], [RequestorID], [RequestorFirstName], [RequestorLastName], [RequestorPhone], [RequestorEmail], [DateTimeSubmitted], [DateTimeStarted], [DateTimeCompleted], [Priority], [RequestType], [RequestTypeStep], [Member], [Status], [Team]) VALUES (25, N',,n,n,n,', N'n,n,', N'A3HR\Lyndon', N'Lyndon', N'Sundmark', N'604 357 3669', N'lvsund@outlook.com', CAST(N'2018-05-19T14:45:01.230' AS DateTime), NULL, NULL, 3, 3, 2, 1, 1, 1)
INSERT [dbo].[ServiceRequests] ([Id], [RequestHeading], [RequestDescription], [RequestorID], [RequestorFirstName], [RequestorLastName], [RequestorPhone], [RequestorEmail], [DateTimeSubmitted], [DateTimeStarted], [DateTimeCompleted], [Priority], [RequestType], [RequestTypeStep], [Member], [Status], [Team]) VALUES (26, N'gdgdg', N'dgdg', N'A3HR\Lyndon', N'Lyndon', N'Sundmark', N'604 357 3669', N'lvsund@outlook.com', CAST(N'2018-05-19T14:58:05.627' AS DateTime), NULL, NULL, 3, 3, 2, 1, 1, 1)
INSERT [dbo].[ServiceRequests] ([Id], [RequestHeading], [RequestDescription], [RequestorID], [RequestorFirstName], [RequestorLastName], [RequestorPhone], [RequestorEmail], [DateTimeSubmitted], [DateTimeStarted], [DateTimeCompleted], [Priority], [RequestType], [RequestTypeStep], [Member], [Status], [Team]) VALUES (28, N'fbfb', N'bffbf', N'A3HR\Lyndon', N'Lyndon', N'Sundmark', N'604 357 3669', N'lvsund@outlook.com', CAST(N'2018-05-20T13:47:03.267' AS DateTime), CAST(N'2018-05-20T13:55:27.787' AS DateTime), NULL, 3, 3, 2, 1, 2, 1)
INSERT [dbo].[ServiceRequests] ([Id], [RequestHeading], [RequestDescription], [RequestorID], [RequestorFirstName], [RequestorLastName], [RequestorPhone], [RequestorEmail], [DateTimeSubmitted], [DateTimeStarted], [DateTimeCompleted], [Priority], [RequestType], [RequestTypeStep], [Member], [Status], [Team]) VALUES (29, N'adada', N'adadad', N'A3HR\Lyndon', N'Lyndon', N'Sundmark', N'604 357 3669', N'lvsund@outlook.com', CAST(N'2018-05-20T14:02:40.763' AS DateTime), NULL, NULL, 3, 3, 2, 1, 1, 1)
INSERT [dbo].[ServiceRequests] ([Id], [RequestHeading], [RequestDescription], [RequestorID], [RequestorFirstName], [RequestorLastName], [RequestorPhone], [RequestorEmail], [DateTimeSubmitted], [DateTimeStarted], [DateTimeCompleted], [Priority], [RequestType], [RequestTypeStep], [Member], [Status], [Team]) VALUES (30, N'fsfsfs', N'sfsf', N'A3HR\Lyndon', N'Lyndon', N'Sundmark', N'604 357 3669', N'lvsund@outlook.com', CAST(N'2018-05-20T14:04:18.837' AS DateTime), NULL, NULL, 3, 3, 2, 1, 1, 1)
INSERT [dbo].[ServiceRequests] ([Id], [RequestHeading], [RequestDescription], [RequestorID], [RequestorFirstName], [RequestorLastName], [RequestorPhone], [RequestorEmail], [DateTimeSubmitted], [DateTimeStarted], [DateTimeCompleted], [Priority], [RequestType], [RequestTypeStep], [Member], [Status], [Team]) VALUES (31, N'gmmgm', N'gmgmg', N'A3HR\Lyndon', N'Lyndon', N'Sundmark', N'604 357 3669', N'lvsund@outlook.com', CAST(N'2018-05-20T14:05:47.037' AS DateTime), NULL, NULL, NULL, 3, 2, 1, 1, 1)
INSERT [dbo].[ServiceRequests] ([Id], [RequestHeading], [RequestDescription], [RequestorID], [RequestorFirstName], [RequestorLastName], [RequestorPhone], [RequestorEmail], [DateTimeSubmitted], [DateTimeStarted], [DateTimeCompleted], [Priority], [RequestType], [RequestTypeStep], [Member], [Status], [Team]) VALUES (32, N'hfhfh', N'fhfh', N'A3HR\Lyndon', N'Lyndon', N'Sundmark', N'604 357 3669', N'lvsund@outlook.com', CAST(N'2018-05-20T14:06:41.970' AS DateTime), NULL, NULL, 3, 3, 2, 2, 1, 2)
INSERT [dbo].[ServiceRequests] ([Id], [RequestHeading], [RequestDescription], [RequestorID], [RequestorFirstName], [RequestorLastName], [RequestorPhone], [RequestorEmail], [DateTimeSubmitted], [DateTimeStarted], [DateTimeCompleted], [Priority], [RequestType], [RequestTypeStep], [Member], [Status], [Team]) VALUES (33, N'tytyy', N'tyty', N'A3HR\Lyndon', N'Lyndon', N'Sundmark', N'604 357 3669', N'lvsund@outlook.com', CAST(N'2018-05-20T14:13:58.517' AS DateTime), NULL, NULL, NULL, NULL, 2, 1, 1, 1)
INSERT [dbo].[ServiceRequests] ([Id], [RequestHeading], [RequestDescription], [RequestorID], [RequestorFirstName], [RequestorLastName], [RequestorPhone], [RequestorEmail], [DateTimeSubmitted], [DateTimeStarted], [DateTimeCompleted], [Priority], [RequestType], [RequestTypeStep], [Member], [Status], [Team]) VALUES (34, N'tutt', N'tut', N'A3HR\Lyndon', N'Lyndon', N'Sundmark', N'604 357 3669', N'lvsund@outlook.com', CAST(N'2018-05-20T14:26:01.070' AS DateTime), NULL, NULL, 3, 3, 2, 1, 1, 1)
INSERT [dbo].[ServiceRequests] ([Id], [RequestHeading], [RequestDescription], [RequestorID], [RequestorFirstName], [RequestorLastName], [RequestorPhone], [RequestorEmail], [DateTimeSubmitted], [DateTimeStarted], [DateTimeCompleted], [Priority], [RequestType], [RequestTypeStep], [Member], [Status], [Team]) VALUES (35, N'khk', N'hkhk', N'A3HR\Lyndon', N'Lyndon', N'Sundmark', N'604 357 3669', N'lvsund@outlook.com', CAST(N'2018-05-20T14:52:22.013' AS DateTime), NULL, NULL, 3, 3, 2, 1, 1, 1)
INSERT [dbo].[ServiceRequests] ([Id], [RequestHeading], [RequestDescription], [RequestorID], [RequestorFirstName], [RequestorLastName], [RequestorPhone], [RequestorEmail], [DateTimeSubmitted], [DateTimeStarted], [DateTimeCompleted], [Priority], [RequestType], [RequestTypeStep], [Member], [Status], [Team]) VALUES (36, N'grgr', N'rggrrg', N'A3HR\Lyndon', N'Lyndon', N'Sundmark', N'604 357 3669', N'lvsund@outlook.com', CAST(N'2018-05-20T14:53:55.983' AS DateTime), NULL, NULL, 3, 3, 2, 1, 1, 1)
INSERT [dbo].[ServiceRequests] ([Id], [RequestHeading], [RequestDescription], [RequestorID], [RequestorFirstName], [RequestorLastName], [RequestorPhone], [RequestorEmail], [DateTimeSubmitted], [DateTimeStarted], [DateTimeCompleted], [Priority], [RequestType], [RequestTypeStep], [Member], [Status], [Team]) VALUES (37, N'gjgjg', N'gjgj afafafafafafa', N'A3HR\Lyndon', N'Lyndon', N'Sundmark', N'604 357 3669', N'lvsund@outlook.com', CAST(N'2018-05-20T14:56:31.987' AS DateTime), NULL, NULL, 3, 3, 2, 1, 1, 2)
INSERT [dbo].[ServiceRequests] ([Id], [RequestHeading], [RequestDescription], [RequestorID], [RequestorFirstName], [RequestorLastName], [RequestorPhone], [RequestorEmail], [DateTimeSubmitted], [DateTimeStarted], [DateTimeCompleted], [Priority], [RequestType], [RequestTypeStep], [Member], [Status], [Team]) VALUES (39, N'gddgdg', N'dgdg', N'A3HR\Lyndon', N'Lyndon', N'Sundmark', N'604 357 3669', N'lvsund@outlook.com', CAST(N'2018-05-25T11:07:46.027' AS DateTime), NULL, NULL, 3, 3, 2, 1, 1, 1)
INSERT [dbo].[ServiceRequests] ([Id], [RequestHeading], [RequestDescription], [RequestorID], [RequestorFirstName], [RequestorLastName], [RequestorPhone], [RequestorEmail], [DateTimeSubmitted], [DateTimeStarted], [DateTimeCompleted], [Priority], [RequestType], [RequestTypeStep], [Member], [Status], [Team]) VALUES (42, N'nvnvnvn', N'vnvnvnvnvnvnvn', N'A3HR\Lyndon', N'Lyndon', N'Sundmark', N'604 357 3669', N'lvsund@outlook.com', CAST(N'2018-05-26T09:59:53.827' AS DateTime), NULL, NULL, 3, 3, 2, 1, 1, 2)
INSERT [dbo].[ServiceRequests] ([Id], [RequestHeading], [RequestDescription], [RequestorID], [RequestorFirstName], [RequestorLastName], [RequestorPhone], [RequestorEmail], [DateTimeSubmitted], [DateTimeStarted], [DateTimeCompleted], [Priority], [RequestType], [RequestTypeStep], [Member], [Status], [Team]) VALUES (44, N'vhvhh', N'vhvh', N'A3HR\Lyndon', N'Lyndon', N'Sundmark', N'604 357 3669', N'lvsund@outlook.com', CAST(N'2018-05-26T10:07:31.443' AS DateTime), NULL, NULL, 3, 3, 2, 1, 1, 2)
INSERT [dbo].[ServiceRequests] ([Id], [RequestHeading], [RequestDescription], [RequestorID], [RequestorFirstName], [RequestorLastName], [RequestorPhone], [RequestorEmail], [DateTimeSubmitted], [DateTimeStarted], [DateTimeCompleted], [Priority], [RequestType], [RequestTypeStep], [Member], [Status], [Team]) VALUES (45, N'cbcbbcb', N'cbcbcb', N'A3HR\Lyndon', N'Lyndon', N'Sundmark', N'604 357 3669', N'lvsund@outlook.com', CAST(N'2018-05-26T10:10:09.107' AS DateTime), NULL, NULL, 3, 3, 2, 2, 1, 2)
SET IDENTITY_INSERT [dbo].[ServiceRequests] OFF
SET IDENTITY_INSERT [dbo].[StatusSets] ON 

INSERT [dbo].[StatusSets] ([Id], [StatusDescription], [LastUpdated], [Active], [StatusType]) VALUES (1, N'Not Started', CAST(N'2016-08-27T13:48:49.440' AS DateTime), 1, 1)
INSERT [dbo].[StatusSets] ([Id], [StatusDescription], [LastUpdated], [Active], [StatusType]) VALUES (2, N'In Progress', CAST(N'2016-08-27T13:48:49.443' AS DateTime), 1, 1)
INSERT [dbo].[StatusSets] ([Id], [StatusDescription], [LastUpdated], [Active], [StatusType]) VALUES (3, N'Completed', CAST(N'2016-08-27T13:48:49.443' AS DateTime), 1, 2)
SET IDENTITY_INSERT [dbo].[StatusSets] OFF
SET IDENTITY_INSERT [dbo].[StatusTypes] ON 

INSERT [dbo].[StatusTypes] ([Id], [StatusTypeDescription]) VALUES (1, N'Open')
INSERT [dbo].[StatusTypes] ([Id], [StatusTypeDescription]) VALUES (2, N'Closed')
SET IDENTITY_INSERT [dbo].[StatusTypes] OFF
SET IDENTITY_INSERT [dbo].[StepHistories] ON 

INSERT [dbo].[StepHistories] ([Id], [LastUpdated], [RequestTypeStep], [ServiceRequest]) VALUES (3, CAST(N'2018-05-11T15:18:29.793' AS DateTime), 2, 7)
INSERT [dbo].[StepHistories] ([Id], [LastUpdated], [RequestTypeStep], [ServiceRequest]) VALUES (5, CAST(N'2018-05-11T18:50:30.693' AS DateTime), 2, 9)
INSERT [dbo].[StepHistories] ([Id], [LastUpdated], [RequestTypeStep], [ServiceRequest]) VALUES (7, CAST(N'2018-05-12T07:53:56.903' AS DateTime), 2, 11)
INSERT [dbo].[StepHistories] ([Id], [LastUpdated], [RequestTypeStep], [ServiceRequest]) VALUES (8, CAST(N'2018-05-12T14:12:48.957' AS DateTime), 2, 12)
INSERT [dbo].[StepHistories] ([Id], [LastUpdated], [RequestTypeStep], [ServiceRequest]) VALUES (10, CAST(N'2018-05-19T14:12:28.243' AS DateTime), 2, 22)
INSERT [dbo].[StepHistories] ([Id], [LastUpdated], [RequestTypeStep], [ServiceRequest]) VALUES (11, CAST(N'2018-05-19T14:38:31.377' AS DateTime), 2, 24)
INSERT [dbo].[StepHistories] ([Id], [LastUpdated], [RequestTypeStep], [ServiceRequest]) VALUES (12, CAST(N'2018-05-19T14:45:01.233' AS DateTime), 2, 25)
INSERT [dbo].[StepHistories] ([Id], [LastUpdated], [RequestTypeStep], [ServiceRequest]) VALUES (13, CAST(N'2018-05-19T14:58:05.630' AS DateTime), 2, 26)
INSERT [dbo].[StepHistories] ([Id], [LastUpdated], [RequestTypeStep], [ServiceRequest]) VALUES (15, CAST(N'2018-05-20T13:47:03.270' AS DateTime), 2, 28)
INSERT [dbo].[StepHistories] ([Id], [LastUpdated], [RequestTypeStep], [ServiceRequest]) VALUES (16, CAST(N'2018-05-20T14:02:40.767' AS DateTime), 2, 29)
INSERT [dbo].[StepHistories] ([Id], [LastUpdated], [RequestTypeStep], [ServiceRequest]) VALUES (17, CAST(N'2018-05-20T14:04:18.840' AS DateTime), 2, 30)
INSERT [dbo].[StepHistories] ([Id], [LastUpdated], [RequestTypeStep], [ServiceRequest]) VALUES (18, CAST(N'2018-05-20T14:05:47.040' AS DateTime), 2, 31)
INSERT [dbo].[StepHistories] ([Id], [LastUpdated], [RequestTypeStep], [ServiceRequest]) VALUES (19, CAST(N'2018-05-20T14:06:41.973' AS DateTime), 2, 32)
INSERT [dbo].[StepHistories] ([Id], [LastUpdated], [RequestTypeStep], [ServiceRequest]) VALUES (20, CAST(N'2018-05-20T14:13:58.520' AS DateTime), 2, 33)
INSERT [dbo].[StepHistories] ([Id], [LastUpdated], [RequestTypeStep], [ServiceRequest]) VALUES (21, CAST(N'2018-05-20T14:26:01.073' AS DateTime), 2, 34)
INSERT [dbo].[StepHistories] ([Id], [LastUpdated], [RequestTypeStep], [ServiceRequest]) VALUES (22, CAST(N'2018-05-20T14:52:22.017' AS DateTime), 2, 35)
INSERT [dbo].[StepHistories] ([Id], [LastUpdated], [RequestTypeStep], [ServiceRequest]) VALUES (23, CAST(N'2018-05-20T14:53:55.987' AS DateTime), 2, 36)
INSERT [dbo].[StepHistories] ([Id], [LastUpdated], [RequestTypeStep], [ServiceRequest]) VALUES (24, CAST(N'2018-05-20T14:56:31.990' AS DateTime), 2, 37)
INSERT [dbo].[StepHistories] ([Id], [LastUpdated], [RequestTypeStep], [ServiceRequest]) VALUES (25, CAST(N'2018-05-25T11:07:46.027' AS DateTime), 2, 39)
INSERT [dbo].[StepHistories] ([Id], [LastUpdated], [RequestTypeStep], [ServiceRequest]) VALUES (26, CAST(N'2018-05-26T09:59:53.830' AS DateTime), 2, 42)
INSERT [dbo].[StepHistories] ([Id], [LastUpdated], [RequestTypeStep], [ServiceRequest]) VALUES (27, CAST(N'2018-05-26T10:07:31.443' AS DateTime), 2, 44)
INSERT [dbo].[StepHistories] ([Id], [LastUpdated], [RequestTypeStep], [ServiceRequest]) VALUES (28, CAST(N'2018-05-26T10:10:09.107' AS DateTime), 2, 45)
SET IDENTITY_INSERT [dbo].[StepHistories] OFF
SET IDENTITY_INSERT [dbo].[TeamAssignmentHistories] ON 

INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (3, N'A3HR\Lyndon', CAST(N'2018-05-11T15:18:29.793' AS DateTime), 7, 1)
INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (4, N'A3HR\Lyndon', CAST(N'2018-05-11T15:44:59.897' AS DateTime), 7, 2)
INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (5, N'A3HR\Lyndon', CAST(N'2018-05-11T15:46:15.707' AS DateTime), 7, 1)
INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (6, N'A3HR\Lyndon', CAST(N'2018-05-11T18:21:13.743' AS DateTime), 7, 2)
INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (8, N'A3HR\Lyndon', CAST(N'2018-05-11T18:50:30.690' AS DateTime), 9, 1)
INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (10, N'A3HR\Lyndon', CAST(N'2018-05-12T07:53:56.903' AS DateTime), 11, 1)
INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (11, N'A3HR\Lyndon', CAST(N'2018-05-12T14:12:48.957' AS DateTime), 12, 1)
INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (14, N'A3HR\Lyndon', CAST(N'2018-05-19T10:22:14.850' AS DateTime), 9, 2)
INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (16, N'A3HR\Lyndon', CAST(N'2018-05-19T14:12:28.243' AS DateTime), 22, 1)
INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (17, N'A3HR\Lyndon', CAST(N'2018-05-19T14:38:31.377' AS DateTime), 24, 2)
INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (18, N'A3HR\Lyndon', CAST(N'2018-05-19T14:45:01.230' AS DateTime), 25, 1)
INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (19, N'A3HR\Lyndon', CAST(N'2018-05-19T14:58:05.627' AS DateTime), 26, 1)
INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (21, N'A3HR\Lyndon', CAST(N'2018-05-20T13:47:03.270' AS DateTime), 28, 1)
INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (22, N'A3HR\Lyndon', CAST(N'2018-05-20T14:02:40.767' AS DateTime), 29, 1)
INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (23, N'A3HR\Lyndon', CAST(N'2018-05-20T14:04:18.840' AS DateTime), 30, 1)
INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (24, N'A3HR\Lyndon', CAST(N'2018-05-20T14:05:47.040' AS DateTime), 31, 1)
INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (25, N'A3HR\Lyndon', CAST(N'2018-05-20T14:06:41.970' AS DateTime), 32, 1)
INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (26, N'A3HR\Lyndon', CAST(N'2018-05-20T14:13:58.520' AS DateTime), 33, 1)
INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (27, N'A3HR\Lyndon', CAST(N'2018-05-20T14:26:01.073' AS DateTime), 34, 1)
INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (28, N'A3HR\Lyndon', CAST(N'2018-05-20T14:52:22.013' AS DateTime), 35, 1)
INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (29, N'A3HR\Lyndon', CAST(N'2018-05-20T14:53:55.987' AS DateTime), 36, 1)
INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (30, N'A3HR\Lyndon', CAST(N'2018-05-20T14:56:31.990' AS DateTime), 37, 1)
INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (31, N'A3HR\Lyndon', CAST(N'2018-05-20T15:00:40.597' AS DateTime), 37, 2)
INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (32, N'A3HR\Lyndon', CAST(N'2018-05-25T11:07:46.027' AS DateTime), 39, 1)
INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (33, N'A3HR\Lyndon', CAST(N'2018-05-26T09:59:53.827' AS DateTime), 42, 1)
INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (34, N'A3HR\Lyndon', CAST(N'2018-05-26T10:03:21.893' AS DateTime), 42, 2)
INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (35, N'A3HR\Lyndon', CAST(N'2018-05-26T10:07:31.443' AS DateTime), 44, 2)
INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (36, N'A3HR\Lyndon', CAST(N'2018-05-26T10:10:09.107' AS DateTime), 45, 2)
INSERT [dbo].[TeamAssignmentHistories] ([Id], [AssignedBy], [DateAssigned], [ServiceRequest], [Team]) VALUES (37, N'A3HR\Lyndon', CAST(N'2018-05-26T11:44:08.550' AS DateTime), 32, 2)
SET IDENTITY_INSERT [dbo].[TeamAssignmentHistories] OFF
SET IDENTITY_INSERT [dbo].[TeamMembers] ON 

INSERT [dbo].[TeamMembers] ([Id], [Member], [Team]) VALUES (4, 1, 2)
INSERT [dbo].[TeamMembers] ([Id], [Member], [Team]) VALUES (5, 2, 2)
INSERT [dbo].[TeamMembers] ([Id], [Member], [Team]) VALUES (6, 1, 1)
SET IDENTITY_INSERT [dbo].[TeamMembers] OFF
SET IDENTITY_INSERT [dbo].[Teams] ON 

INSERT [dbo].[Teams] ([Id], [TeamDescription], [TeamEmailAddress]) VALUES (1, N'--UnAssigned--', N'unassigned@unassigned.com')
INSERT [dbo].[Teams] ([Id], [TeamDescription], [TeamEmailAddress]) VALUES (2, N'HR Technology Solutions', N'lvsund@outlook.com')
SET IDENTITY_INSERT [dbo].[Teams] OFF
/****** Object:  Index [IX_FileDetails_ServiceRequestID]    Script Date: 2018-06-11 10:01:08 AM ******/
CREATE NONCLUSTERED INDEX [IX_FileDetails_ServiceRequestID] ON [dbo].[FileDetails]
(
	[ServiceRequestID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO
/****** Object:  Index [IX_IndividualAssignmentHistories_AssignedTo]    Script Date: 2018-06-11 10:01:08 AM ******/
CREATE NONCLUSTERED INDEX [IX_IndividualAssignmentHistories_AssignedTo] ON [dbo].[IndividualAssignmentHistories]
(
	[AssignedTo] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO
/****** Object:  Index [IX_IndividualAssignmentHistories_ServiceRequest]    Script Date: 2018-06-11 10:01:08 AM ******/
CREATE NONCLUSTERED INDEX [IX_IndividualAssignmentHistories_ServiceRequest] ON [dbo].[IndividualAssignmentHistories]
(
	[ServiceRequest] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO
/****** Object:  Index [IX_Priorities_Team]    Script Date: 2018-06-11 10:01:08 AM ******/
CREATE NONCLUSTERED INDEX [IX_Priorities_Team] ON [dbo].[Priorities]
(
	[Team] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO
/****** Object:  Index [IX_RequestTypes_Team]    Script Date: 2018-06-11 10:01:08 AM ******/
CREATE NONCLUSTERED INDEX [IX_RequestTypes_Team] ON [dbo].[RequestTypes]
(
	[Team] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO
/****** Object:  Index [IX_RequestTypeSteps_RequestType]    Script Date: 2018-06-11 10:01:08 AM ******/
CREATE NONCLUSTERED INDEX [IX_RequestTypeSteps_RequestType] ON [dbo].[RequestTypeSteps]
(
	[RequestType] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO
/****** Object:  Index [IX_ServiceRequestNotes_ServiceRequest]    Script Date: 2018-06-11 10:01:08 AM ******/
CREATE NONCLUSTERED INDEX [IX_ServiceRequestNotes_ServiceRequest] ON [dbo].[ServiceRequestNotes]
(
	[ServiceRequest] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO
/****** Object:  Index [IX_ServiceRequests_Member]    Script Date: 2018-06-11 10:01:08 AM ******/
CREATE NONCLUSTERED INDEX [IX_ServiceRequests_Member] ON [dbo].[ServiceRequests]
(
	[Member] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO
/****** Object:  Index [IX_ServiceRequests_Priority]    Script Date: 2018-06-11 10:01:08 AM ******/
CREATE NONCLUSTERED INDEX [IX_ServiceRequests_Priority] ON [dbo].[ServiceRequests]
(
	[Priority] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO
/****** Object:  Index [IX_ServiceRequests_RequestType]    Script Date: 2018-06-11 10:01:08 AM ******/
CREATE NONCLUSTERED INDEX [IX_ServiceRequests_RequestType] ON [dbo].[ServiceRequests]
(
	[RequestType] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO
/****** Object:  Index [IX_ServiceRequests_RequestTypeStep]    Script Date: 2018-06-11 10:01:08 AM ******/
CREATE NONCLUSTERED INDEX [IX_ServiceRequests_RequestTypeStep] ON [dbo].[ServiceRequests]
(
	[RequestTypeStep] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO
/****** Object:  Index [IX_ServiceRequests_Status]    Script Date: 2018-06-11 10:01:08 AM ******/
CREATE NONCLUSTERED INDEX [IX_ServiceRequests_Status] ON [dbo].[ServiceRequests]
(
	[Status] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO
/****** Object:  Index [IX_ServiceRequests_Team]    Script Date: 2018-06-11 10:01:08 AM ******/
CREATE NONCLUSTERED INDEX [IX_ServiceRequests_Team] ON [dbo].[ServiceRequests]
(
	[Team] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO
/****** Object:  Index [IX_StatusSets_StatusType]    Script Date: 2018-06-11 10:01:08 AM ******/
CREATE NONCLUSTERED INDEX [IX_StatusSets_StatusType] ON [dbo].[StatusSets]
(
	[StatusType] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO
/****** Object:  Index [IX_StepHistories_RequestTypeStep]    Script Date: 2018-06-11 10:01:08 AM ******/
CREATE NONCLUSTERED INDEX [IX_StepHistories_RequestTypeStep] ON [dbo].[StepHistories]
(
	[RequestTypeStep] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO
/****** Object:  Index [IX_StepHistories_ServiceRequest]    Script Date: 2018-06-11 10:01:08 AM ******/
CREATE NONCLUSTERED INDEX [IX_StepHistories_ServiceRequest] ON [dbo].[StepHistories]
(
	[ServiceRequest] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO
/****** Object:  Index [IX_TeamAssignmentHistories_ServiceRequest]    Script Date: 2018-06-11 10:01:08 AM ******/
CREATE NONCLUSTERED INDEX [IX_TeamAssignmentHistories_ServiceRequest] ON [dbo].[TeamAssignmentHistories]
(
	[ServiceRequest] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO
/****** Object:  Index [IX_TeamAssignmentHistories_Team]    Script Date: 2018-06-11 10:01:08 AM ******/
CREATE NONCLUSTERED INDEX [IX_TeamAssignmentHistories_Team] ON [dbo].[TeamAssignmentHistories]
(
	[Team] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO
/****** Object:  Index [IX_TeamMembers_Member]    Script Date: 2018-06-11 10:01:08 AM ******/
CREATE NONCLUSTERED INDEX [IX_TeamMembers_Member] ON [dbo].[TeamMembers]
(
	[Member] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO
/****** Object:  Index [IX_TeamMembers_Team]    Script Date: 2018-06-11 10:01:08 AM ******/
CREATE NONCLUSTERED INDEX [IX_TeamMembers_Team] ON [dbo].[TeamMembers]
(
	[Team] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO
ALTER TABLE [dbo].[FileDetails]  WITH CHECK ADD  CONSTRAINT [FK_FileDetail_ServiceRequests] FOREIGN KEY([ServiceRequestID])
REFERENCES [dbo].[ServiceRequests] ([Id])
ON DELETE CASCADE
GO
ALTER TABLE [dbo].[FileDetails] CHECK CONSTRAINT [FK_FileDetail_ServiceRequests]
GO
ALTER TABLE [dbo].[IndividualAssignmentHistories]  WITH CHECK ADD  CONSTRAINT [FK_dbo_IndividualAssignmentHistories_dbo_ServiceRequests_ServiceRequest] FOREIGN KEY([ServiceRequest])
REFERENCES [dbo].[ServiceRequests] ([Id])
ON DELETE CASCADE
GO
ALTER TABLE [dbo].[IndividualAssignmentHistories] CHECK CONSTRAINT [FK_dbo_IndividualAssignmentHistories_dbo_ServiceRequests_ServiceRequest]
GO
ALTER TABLE [dbo].[IndividualAssignmentHistories]  WITH CHECK ADD  CONSTRAINT [FK_IndividualAssignmentHistories_Members] FOREIGN KEY([AssignedTo])
REFERENCES [dbo].[Members] ([Id])
GO
ALTER TABLE [dbo].[IndividualAssignmentHistories] CHECK CONSTRAINT [FK_IndividualAssignmentHistories_Members]
GO
ALTER TABLE [dbo].[Priorities]  WITH CHECK ADD  CONSTRAINT [FK_dbo_Priorities_dbo_Teams_Team] FOREIGN KEY([Team])
REFERENCES [dbo].[Teams] ([Id])
GO
ALTER TABLE [dbo].[Priorities] CHECK CONSTRAINT [FK_dbo_Priorities_dbo_Teams_Team]
GO
ALTER TABLE [dbo].[RequestTypes]  WITH CHECK ADD  CONSTRAINT [FK_dbo_RequestTypes_dbo_Teams_Team] FOREIGN KEY([Team])
REFERENCES [dbo].[Teams] ([Id])
GO
ALTER TABLE [dbo].[RequestTypes] CHECK CONSTRAINT [FK_dbo_RequestTypes_dbo_Teams_Team]
GO
ALTER TABLE [dbo].[RequestTypeSteps]  WITH CHECK ADD  CONSTRAINT [FK_dbo_RequestTypeSteps_dbo_RequestTypes_RequestType] FOREIGN KEY([RequestType])
REFERENCES [dbo].[RequestTypes] ([Id])
GO
ALTER TABLE [dbo].[RequestTypeSteps] CHECK CONSTRAINT [FK_dbo_RequestTypeSteps_dbo_RequestTypes_RequestType]
GO
ALTER TABLE [dbo].[ServiceRequestNotes]  WITH CHECK ADD  CONSTRAINT [FK_ServiceRequestNotes_ServiceRequests] FOREIGN KEY([ServiceRequest])
REFERENCES [dbo].[ServiceRequests] ([Id])
ON DELETE CASCADE
GO
ALTER TABLE [dbo].[ServiceRequestNotes] CHECK CONSTRAINT [FK_ServiceRequestNotes_ServiceRequests]
GO
ALTER TABLE [dbo].[ServiceRequests]  WITH CHECK ADD  CONSTRAINT [FK_dbo_ServiceRequests_dbo_Members_Member] FOREIGN KEY([Member])
REFERENCES [dbo].[Members] ([Id])
GO
ALTER TABLE [dbo].[ServiceRequests] CHECK CONSTRAINT [FK_dbo_ServiceRequests_dbo_Members_Member]
GO
ALTER TABLE [dbo].[ServiceRequests]  WITH CHECK ADD  CONSTRAINT [FK_dbo_ServiceRequests_dbo_Priorities_Priority] FOREIGN KEY([Priority])
REFERENCES [dbo].[Priorities] ([Id])
GO
ALTER TABLE [dbo].[ServiceRequests] CHECK CONSTRAINT [FK_dbo_ServiceRequests_dbo_Priorities_Priority]
GO
ALTER TABLE [dbo].[ServiceRequests]  WITH CHECK ADD  CONSTRAINT [FK_dbo_ServiceRequests_dbo_RequestTypes_RequestType] FOREIGN KEY([RequestType])
REFERENCES [dbo].[RequestTypes] ([Id])
GO
ALTER TABLE [dbo].[ServiceRequests] CHECK CONSTRAINT [FK_dbo_ServiceRequests_dbo_RequestTypes_RequestType]
GO
ALTER TABLE [dbo].[ServiceRequests]  WITH CHECK ADD  CONSTRAINT [FK_dbo_ServiceRequests_dbo_RequestTypeSteps_RequestTypeStep] FOREIGN KEY([RequestTypeStep])
REFERENCES [dbo].[RequestTypeSteps] ([Id])
GO
ALTER TABLE [dbo].[ServiceRequests] CHECK CONSTRAINT [FK_dbo_ServiceRequests_dbo_RequestTypeSteps_RequestTypeStep]
GO
ALTER TABLE [dbo].[ServiceRequests]  WITH CHECK ADD  CONSTRAINT [FK_dbo_ServiceRequests_dbo_StatusSets_Status] FOREIGN KEY([Status])
REFERENCES [dbo].[StatusSets] ([Id])
GO
ALTER TABLE [dbo].[ServiceRequests] CHECK CONSTRAINT [FK_dbo_ServiceRequests_dbo_StatusSets_Status]
GO
ALTER TABLE [dbo].[ServiceRequests]  WITH CHECK ADD  CONSTRAINT [FK_dbo_ServiceRequests_dbo_Teams_Team] FOREIGN KEY([Team])
REFERENCES [dbo].[Teams] ([Id])
GO
ALTER TABLE [dbo].[ServiceRequests] CHECK CONSTRAINT [FK_dbo_ServiceRequests_dbo_Teams_Team]
GO
ALTER TABLE [dbo].[StatusSets]  WITH CHECK ADD  CONSTRAINT [FK_dbo_StatusSets_dbo_StatusTypes_StatusType] FOREIGN KEY([StatusType])
REFERENCES [dbo].[StatusTypes] ([Id])
GO
ALTER TABLE [dbo].[StatusSets] CHECK CONSTRAINT [FK_dbo_StatusSets_dbo_StatusTypes_StatusType]
GO
ALTER TABLE [dbo].[StepHistories]  WITH CHECK ADD  CONSTRAINT [FK_dbo_StepHistories_dbo_RequestTypeSteps_RequestTypeStep] FOREIGN KEY([RequestTypeStep])
REFERENCES [dbo].[RequestTypeSteps] ([Id])
GO
ALTER TABLE [dbo].[StepHistories] CHECK CONSTRAINT [FK_dbo_StepHistories_dbo_RequestTypeSteps_RequestTypeStep]
GO
ALTER TABLE [dbo].[StepHistories]  WITH CHECK ADD  CONSTRAINT [FK_dbo_StepHistories_dbo_ServiceRequests_ServiceRequest] FOREIGN KEY([ServiceRequest])
REFERENCES [dbo].[ServiceRequests] ([Id])
ON DELETE CASCADE
GO
ALTER TABLE [dbo].[StepHistories] CHECK CONSTRAINT [FK_dbo_StepHistories_dbo_ServiceRequests_ServiceRequest]
GO
ALTER TABLE [dbo].[TeamAssignmentHistories]  WITH CHECK ADD  CONSTRAINT [FK_dbo_TeamAssignmentHistories_dbo_ServiceRequests_ServiceRequest] FOREIGN KEY([ServiceRequest])
REFERENCES [dbo].[ServiceRequests] ([Id])
ON DELETE CASCADE
GO
ALTER TABLE [dbo].[TeamAssignmentHistories] CHECK CONSTRAINT [FK_dbo_TeamAssignmentHistories_dbo_ServiceRequests_ServiceRequest]
GO
ALTER TABLE [dbo].[TeamAssignmentHistories]  WITH CHECK ADD  CONSTRAINT [FK_dbo_TeamAssignmentHistories_dbo_Teams_Team] FOREIGN KEY([Team])
REFERENCES [dbo].[Teams] ([Id])
ON DELETE CASCADE
GO
ALTER TABLE [dbo].[TeamAssignmentHistories] CHECK CONSTRAINT [FK_dbo_TeamAssignmentHistories_dbo_Teams_Team]
GO
ALTER TABLE [dbo].[TeamMembers]  WITH CHECK ADD  CONSTRAINT [FK_dbo_TeamMembers_dbo_Members_Member] FOREIGN KEY([Member])
REFERENCES [dbo].[Members] ([Id])
GO
ALTER TABLE [dbo].[TeamMembers] CHECK CONSTRAINT [FK_dbo_TeamMembers_dbo_Members_Member]
GO
ALTER TABLE [dbo].[TeamMembers]  WITH CHECK ADD  CONSTRAINT [FK_dbo_TeamMembers_dbo_Teams_Team] FOREIGN KEY([Team])
REFERENCES [dbo].[Teams] ([Id])
GO
ALTER TABLE [dbo].[TeamMembers] CHECK CONSTRAINT [FK_dbo_TeamMembers_dbo_Teams_Team]
GO
USE [master]
GO
ALTER DATABASE [ServeMeHRCore] SET  READ_WRITE 
GO
