-- MySQL dump 10.13  Distrib 8.0.33, for Win64 (x86_64)
--
-- Host: localhost    Database: FitMyVibe
-- ------------------------------------------------------
-- Server version	8.0.33

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `proddesc`
--

DROP TABLE IF EXISTS `proddesc`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `proddesc` (
  `prodID` int NOT NULL,
  `prodName` varchar(255) NOT NULL,
  `prodGender` int DEFAULT NULL,
  `prodComp` varchar(255) DEFAULT NULL,
  `prodColor` varchar(255) DEFAULT NULL,
  `prodCategory` varchar(255) DEFAULT NULL,
  `prodOrigin` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`prodID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `prodlink`
--

DROP TABLE IF EXISTS `prodlink`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `prodlink` (
  `prodID` int DEFAULT NULL,
  `prodLink` varchar(255) DEFAULT NULL,
  `prodImageLink` varchar(255) DEFAULT NULL,
  KEY `prodID` (`prodID`),
  CONSTRAINT `prodlink_ibfk_1` FOREIGN KEY (`prodID`) REFERENCES `proddesc` (`prodID`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `prodprice`
--

DROP TABLE IF EXISTS `prodprice`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `prodprice` (
  `prodID` int DEFAULT NULL,
  `prodInStock` tinyint(1) DEFAULT NULL,
  `prodRegPrice` decimal(5,2) DEFAULT NULL,
  `prodSalePrice` decimal(5,2) DEFAULT NULL,
  `prodDiscPrice` decimal(5,2) DEFAULT NULL,
  KEY `prodID` (`prodID`),
  CONSTRAINT `prodprice_ibfk_1` FOREIGN KEY (`prodID`) REFERENCES `proddesc` (`prodID`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `prodvec`
--

DROP TABLE IF EXISTS `prodvec`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `prodvec` (
  `prodID` int NOT NULL,
  `vecPos` int NOT NULL,
  `vecVal` decimal(10,8) DEFAULT NULL,
  PRIMARY KEY (`prodID`,`vecPos`),
  CONSTRAINT `prodvec_ibfk_1` FOREIGN KEY (`prodID`) REFERENCES `proddesc` (`prodID`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-04-14 17:01:18
